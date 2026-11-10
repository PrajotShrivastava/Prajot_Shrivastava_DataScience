#!/usr/bin/env python3
"""
Small evaluation of the RAG prototype.

1. Retrieval quality on 17 hand-labelled answerable questions, for 4 retrievers
   (BM25, dense, hybrid RRF, hybrid + cross-encoder rerank) at k = 1, 3, 5:
     doc_precision@k  = share of the top-k passages that come from a relevant document
     doc_recall@k     = share of the relevant documents that appear in the top-k
     hit@k            = 1 if any top-k passage is a *relevant passage* (relevant doc + key phrase)
     MRR@10           = 1 / rank of the first relevant passage (0 if none in top 10)
2. Guardrails: out-of-scope questions must hit the no-answer fallback, sensitive questions must be
   blocked, answerable questions must NOT be refused (false-refusal rate).
3. (optional, --llm qwen|flan-t5|extractive) end-to-end answers: citation rate, supported-sentence
   fraction (faithfulness proxy), key-phrase coverage, latency.

Outputs: evaluation.csv (per question x method x k), evaluation_summary.csv, guardrail_eval.csv,
         end_to_end_eval.csv + eval_answers.jsonl (if --llm given)
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import rag_prototype as R

HERE = Path(__file__).resolve().parent
METHODS = ["bm25", "dense", "hybrid", "hybrid+rerank"]
KS = [1, 3, 5]


def is_rel_chunk(ch, q) -> bool:
    t = ch.text.lower()
    return ch.doc in q["relevant_docs"] and any(k.lower() in t for k in q["key_phrases"])


def retrieval_eval(rag: R.RAG, qs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    chunks = rag.retriever.chunks
    for q in qs:
        n_rel = sum(is_rel_chunk(c, q) for c in chunks)
        if n_rel == 0:
            print(f"WARNING {q['id']}: no chunk in the index matches its key phrases")
        for m in METHODS:
            hits = rag.retriever.search(q["question"], m, k=10)
            ids = [i for i, _ in hits]
            rel = [is_rel_chunk(chunks[i], q) for i in ids]
            first = next((r + 1 for r, v in enumerate(rel) if v), None)
            for k in KS:
                top = ids[:k]
                docs = [chunks[i].doc for i in top]
                rows.append({
                    "qid": q["id"], "method": m, "k": k,
                    "doc_precision": np.mean([d in q["relevant_docs"] for d in docs]) if docs else 0.0,
                    "doc_recall": len(set(docs) & set(q["relevant_docs"])) / len(q["relevant_docs"]),
                    "hit": float(any(rel[:k])),
                    "mrr@10": 1.0 / first if first else 0.0,
                    "n_relevant_passages_in_index": n_rel,
                    "top1": f"{chunks[ids[0]].doc} p{chunks[ids[0]].page}" if ids else "",
                })
    df = pd.DataFrame(rows)
    summ = (df.groupby(["method", "k"])[["doc_precision", "doc_recall", "hit", "mrr@10"]].mean()
            .round(3).reset_index().rename(columns={"doc_precision": "precision@k (doc)", "doc_recall": "recall@k (doc)",
                                                    "hit": "hit@k (passage)", "mrr@10": "MRR@10 (passage)"}))
    summ["method"] = pd.Categorical(summ["method"], METHODS, ordered=True)
    return df, summ.sort_values(["k", "method"])


def gate_only(rag: R.RAG, q: str) -> str:
    """Status the pipeline would reach before generation (input guardrail + relevance gate)."""
    blocked, _ = R.input_guardrail(q)
    if blocked:
        return "blocked"
    hits = rag.retriever.search(q, "hybrid+rerank", R.TOP_K)
    if not hits or hits[0][1] < R.RELATED_MIN_SCORE:
        return "no_answer"
    return "related_only" if hits[0][1] < R.RERANK_MIN_SCORE else "passes_to_llm"


def guardrail_eval(rag: R.RAG, data: dict) -> pd.DataFrame:
    rows = []
    for group, expect in [("answerable", "passes_to_llm"), ("out_of_scope", "no_answer"), ("sensitive", "blocked")]:
        for q in data[group]:
            hits = rag.retriever.search(q["question"], "hybrid+rerank", 1)
            got = gate_only(rag, q["question"])
            rows.append({"qid": q["id"], "group": group, "question": q["question"], "expected": expect, "got": got,
                         "correct": got == expect, "top_rerank_score": round(hits[0][1], 3) if hits else None})
    return pd.DataFrame(rows)


def e2e_eval(rag: R.RAG, data: dict, out: Path) -> pd.DataFrame:
    rows = []
    with open(out / "eval_answers.jsonl", "w") as fh:
        for group in ["answerable", "out_of_scope", "sensitive"]:
            for q in data[group]:
                a = rag.ask(q["question"])
                fh.write(json.dumps({"qid": q["id"], "group": group, **asdict(a)}) + "\n")
                frac = next((float(f.split("=")[1]) for f in a.guardrails if f.startswith("info:supported")), None)
                cited_docs = {c["doc"] for c in a.citations}
                rows.append({
                    "qid": q["id"], "group": group, "status": a.status,
                    "has_citation": bool(a.citations),
                    "cites_relevant_doc": bool(cited_docs & set(q.get("relevant_docs", []))) if group == "answerable" else None,
                    "llm_draft_supported_fraction": frac,
                    "key_phrase_in_answer": any(k.lower() in a.answer.lower() for k in q.get("key_phrases", []))
                    if group == "answerable" else None,
                    "fallbacks": ";".join(f for f in a.guardrails if "->" in f or f.startswith("fallback")),
                    "latency_s": a.timings_s.get("total"),
                })
                print(f"  {q['id']} {a.status:15s} cites={len(a.citations)} {a.timings_s}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["qwen", "flan-t5", "extractive"], default=None,
                    help="also run the full pipeline end-to-end with this generator")
    args = ap.parse_args()
    data = json.loads((HERE / "eval_questions.json").read_text())
    rag = R.RAG(args.llm or "extractive")

    per_q, summ = retrieval_eval(rag, data["answerable"])
    per_q.to_csv(HERE / "evaluation.csv", index=False)
    summ.to_csv(HERE / "evaluation_summary.csv", index=False)
    print("\nRetrieval quality (mean over %d answerable questions):" % len(data["answerable"]))
    print(summ.to_string(index=False))

    g = guardrail_eval(rag, data)
    g.to_csv(HERE / "guardrail_eval.csv", index=False)
    print("\nGuardrails (gate-level, no LLM):")
    print(g.groupby("group").correct.agg(["mean", "sum", "count"]).rename(columns={"mean": "accuracy"}).to_string())
    wrong = g[~g.correct]
    if len(wrong):
        print("Misclassified:\n" + wrong[["qid", "expected", "got", "top_rerank_score"]].to_string(index=False))

    if args.llm:
        print(f"\nEnd-to-end with generator = {args.llm}:")
        e = e2e_eval(rag, data, HERE)
        e.to_csv(HERE / "end_to_end_eval.csv", index=False)
        a = e[e.group == "answerable"]
        summary = {
            "answered (LLM or extractive)": a.status.isin(["answered", "answered_extractive"]).mean(),
            "answered by LLM after NLI filtering": (a.status == "answered").mean(),
            "fell back to extractive quotes": (a.status == "answered_extractive").mean(),
            "citation rate": a.has_citation.mean(),
            "cites a relevant document": a.cites_relevant_doc.mean(),
            "LLM-draft sentences entailed by sources (before filtering)": a.llm_draft_supported_fraction.mean(),
            "key phrase in final answer": a.key_phrase_in_answer.mean(),
            "out-of-scope correctly refused": (e[e.group == "out_of_scope"].status == "no_answer").mean(),
            "sensitive correctly blocked": (e[e.group == "sensitive"].status == "blocked").mean(),
            "median latency answerable (s)": a.latency_s.median(),
        }
        pd.Series(summary).round(3).rename("value").to_csv(HERE / "end_to_end_summary.csv", header=True)
        for k_, v_ in summary.items():
            print(f"  {k_:60s} {v_:.2f}")


if __name__ == "__main__":
    main()
