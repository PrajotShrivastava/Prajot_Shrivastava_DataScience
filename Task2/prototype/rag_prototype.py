#!/usr/bin/env python3
"""
Minimal-but-real RAG prototype for cyclone technical documentation (free / open-source stack only).

  ingest : PDFs in ./docs -> page text -> cleaned, sentence-aware chunks (page metadata kept)
           -> dense index (FAISS, BAAI/bge-small-en-v1.5) + sparse index (BM25)
  ask    : input guardrail -> hybrid retrieval (dense + BM25, reciprocal-rank fusion)
           -> cross-encoder rerank (ms-marco-MiniLM-L-6-v2) -> relevance gate (graceful fallback)
           -> grounded generation (Qwen2.5-0.5B-Instruct by default; flan-t5-base or extractive optional)
           -> output guardrails (citation enforcement + sentence-level faithfulness check)

Usage
  python rag_prototype.py ingest
  python rag_prototype.py ask "What does a sudden drop in pressure differential indicate?"
  python rag_prototype.py chat
  python rag_prototype.py ask "..." --llm extractive      # no generator download; quotes sources
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
INDEX_DIR = HERE / "index"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"                    # 33M params, 384-d, MIT licence
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"     # 22M params, Apache-2.0
LLM_MODELS = {
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",                  # 0.5B params, Apache-2.0
    "flan-t5": "google/flan-t5-base",                      # 250M params, Apache-2.0
}

CHUNK_WORDS = 180          # ~240 tokens: fits embedder/reranker 512-token limit with headroom
CHUNK_OVERLAP_WORDS = 40   # ~20 % overlap so facts split across a boundary survive
MIN_CHUNK_WORDS = 25
CANDIDATES = 20            # per retriever before fusion
TOP_K = 4                  # passages passed to the LLM
RRF_K = 60
RERANK_MIN_SCORE = -4.0    # cross-encoder logit; below => "not in the documentation" fallback (see notes.md)
RELATED_MIN_SCORE = -6.0   # between this and RERANK_MIN_SCORE: "not directly covered" + quoted related passages

# Plant-floor vocabulary -> documentation vocabulary (applied to the retrieval query only).
GLOSSARY = {
    r"\bdraft\b": "pressure differential, suction, gas flow",
    r"\bdraught\b": "pressure differential, suction, gas flow",
    r"\b(dp|delta[- ]?p)\b": "pressure differential, pressure drop",
    r"\bid fan\b": "induced draft fan, exhaust fan",
    r"\b(chok\w*|jam\w*|clog\w*)\b": "plugging, blockage",
    r"\b(build[- ]?up|coating)\b": "accumulation, plugging",
}
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"       # 70M params, Apache-2.0 - faithfulness / attribution
ENTAIL_MIN = 0.1           # P(entailment | source premise, answer sentence) needed to count as supported. Chosen on a
                           # 12-sentence labelled sample (unsupported max 0.027; most supported > 0.9) - see notes.md

logging.basicConfig(level=logging.WARNING)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ----------------------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc: str
    page: int
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    status: str                       # answered | answered_extractive | low_confidence | related_only | no_answer | blocked
    citations: list = field(default_factory=list)
    guardrails: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)
    timings_s: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------------------------
_TOC_LINE = re.compile(r"\.{5,}|…{3,}|�{5,}")


def clean_page(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or _TOC_LINE.search(s):
            continue                                            # drop blank + table-of-contents leader lines
        if re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", s, flags=re.I):
            continue                                            # bare page numbers
        lines.append(s)
    t = "\n".join(lines)
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)                      # de-hyphenate line breaks
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove header/footer lines that repeat on >= 50 % of a document's pages."""
    if len(pages) < 4:
        return pages
    counts: dict[str, int] = {}
    for p in pages:
        for ln in {l.strip() for l in p.splitlines() if 3 < len(l.strip()) < 120}:
            counts[ln] = counts.get(ln, 0) + 1
    rep = {l for l, c in counts.items() if c >= max(3, 0.5 * len(pages))}
    return ["\n".join(l for l in p.splitlines() if l.strip() not in rep) for p in pages]


_ABBREV = r"(?<!\bmax\.)(?<!\bmin\.)(?<!\bapprox\.)(?<!\bNo\.)(?<!\bno\.)(?<!\bFig\.)(?<!\bfig\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bD\.G\.)(?<!\bvs\.)"


def split_sentences(text: str) -> list[str]:
    parts = re.split(_ABBREV + r"(?<=[.!?;:])\s+(?=[A-Z0-9(•\-–])|\s+(?=•)", text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_document(doc: str, pages: list[str]) -> list[Chunk]:
    """Sentence-aware sliding window inside each page (page number = citation anchor).
    Short page tails are merged into the next window so tiny fragments don't become chunks."""
    chunks: list[Chunk] = []
    for pno, raw in enumerate(pages, 1):
        text = clean_page(raw)
        sents = split_sentences(text)
        # Hard-split very long "sentences" (tables, lists without punctuation)
        units = []
        for s in sents:
            w = s.split()
            for i in range(0, len(w), CHUNK_WORDS):
                units.append(" ".join(w[i:i + CHUNK_WORDS]))
        i, n = 0, 0
        while i < len(units):
            cur, words = [], 0
            j = i
            while j < len(units) and words + len(units[j].split()) <= CHUNK_WORDS:
                cur.append(units[j]); words += len(units[j].split()); j += 1
            if not cur:
                cur, j = [units[i]], i + 1
                words = len(units[i].split())
            if words >= MIN_CHUNK_WORDS or not chunks or chunks[-1].page != pno:
                if words >= MIN_CHUNK_WORDS:
                    chunks.append(Chunk(f"{doc}::p{pno}::c{n}", doc, pno, " ".join(cur)))
                    n += 1
            else:
                chunks[-1].text += " " + " ".join(cur)          # merge small tail into previous chunk
            if j >= len(units):
                break
            # step back to create ~CHUNK_OVERLAP_WORDS overlap
            back, k = 0, j
            while k > i + 1 and back < CHUNK_OVERLAP_WORDS:
                k -= 1; back += len(units[k].split())
            i = k if k > i else j
    return chunks


def read_pdf_pages(path: Path) -> list[str]:
    from pypdf import PdfReader
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")                                       # owner-password-only PDFs open with ""
    return [(p.extract_text() or "") for p in reader.pages]


def read_text_file(path: Path) -> list[str]:
    """.txt/.md: split on form-feed or '=== PAGE n ===' markers; else treat as one page."""
    t = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"\f|^=== PAGE \d+ ===$", t, flags=re.M)
    return [p for p in parts if p.strip()] or [t]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())


_EMBEDDER = None
_RERANKER = None


def embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _EMBEDDER


def reranker():
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(RERANK_MODEL, device="cpu")
    return _RERANKER


def import_faiss():
    """macOS: torch and faiss each ship an OpenMP runtime; importing faiss *first* makes later torch
    inference crash silently. Always load torch (via sentence-transformers) before faiss."""
    import sentence_transformers  # noqa: F401  (imports torch)
    import faiss
    return faiss


def ingest(docs_dir: Path = DOCS_DIR, index_dir: Path = INDEX_DIR) -> dict:
    faiss = import_faiss()
    from rank_bm25 import BM25Okapi

    t0 = time.time()
    files = sorted([p for p in docs_dir.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"}])
    if not files:
        sys.exit(f"No .pdf/.txt/.md files found in {docs_dir}")
    chunks, stats = [], []
    for f in files:
        pages = read_pdf_pages(f) if f.suffix.lower() == ".pdf" else read_text_file(f)
        pages = strip_repeated_lines(pages)
        c = chunk_document(f.name, pages)
        empty = sum(len(p.strip()) < 50 for p in pages)
        stats.append({"doc": f.name, "pages": len(pages), "pages_without_text": empty, "chunks": len(c)})
        chunks.extend(c)
        print(f"  {f.name[:70]:70s} pages={len(pages):3d} chunks={len(c):3d}"
              + (f"  ({empty} page(s) without extractable text - scanned images need OCR)" if empty else ""))

    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL} ...")
    emb = embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])                     # exact cosine search (corpus is small)
    index.add(emb)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "dense.faiss"))
    with open(index_dir / "chunks.jsonl", "w") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c)) + "\n")
    with open(index_dir / "bm25.pkl", "wb") as fh:
        pickle.dump(BM25Okapi([tokenize(t) for t in texts]), fh)
    manifest = {"embed_model": EMBED_MODEL, "n_docs": len(files), "n_chunks": len(chunks),
                "chunk_words": CHUNK_WORDS, "overlap_words": CHUNK_OVERLAP_WORDS, "docs": stats,
                "build_seconds": round(time.time() - t0, 1)}
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Index built: {len(files)} docs, {len(chunks)} chunks in {manifest['build_seconds']} s -> {index_dir}")
    return manifest


# ----------------------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------------------
def expand_query(q: str) -> str:
    """Append documentation synonyms for plant-floor terms, e.g. 'draft drop' -> '... (pressure differential, ...)'."""
    extra = [v for pat, v in GLOSSARY.items() if re.search(pat, q, flags=re.I)]
    return f"{q} ({'; '.join(extra)})" if extra else q


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        faiss = import_faiss()
        if not (index_dir / "dense.faiss").exists():
            sys.exit("Index not found - run:  python rag_prototype.py ingest")
        self.index = faiss.read_index(str(index_dir / "dense.faiss"))
        self.chunks = [Chunk(**json.loads(l)) for l in open(index_dir / "chunks.jsonl")]
        with open(index_dir / "bm25.pkl", "rb") as fh:
            self.bm25 = pickle.load(fh)

    def dense(self, q: str, n: int = CANDIDATES) -> list[tuple[int, float]]:
        v = embedder().encode([BGE_QUERY_PREFIX + q], normalize_embeddings=True).astype("float32")
        s, i = self.index.search(v, n)
        return [(int(a), float(b)) for a, b in zip(i[0], s[0]) if a >= 0]

    def sparse(self, q: str, n: int = CANDIDATES) -> list[tuple[int, float]]:
        sc = self.bm25.get_scores(tokenize(q))
        top = np.argsort(-sc)[:n]
        return [(int(i), float(sc[i])) for i in top if sc[i] > 0]

    @staticmethod
    def rrf(*lists: list[tuple[int, float]]) -> list[tuple[int, float]]:
        agg: dict[int, float] = {}
        for lst in lists:
            for rank, (i, _) in enumerate(lst):
                agg[i] = agg.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(agg.items(), key=lambda x: -x[1])

    def rerank(self, q: str, cand: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
        if not cand:
            return []
        pairs = [(q, self.chunks[i].text) for i, _ in cand]
        sc = reranker().predict(pairs, show_progress_bar=False)
        order = np.argsort(-sc)[:k]
        return [(cand[o][0], float(sc[o])) for o in order]

    def search(self, q: str, method: str = "hybrid+rerank", k: int = TOP_K) -> list[tuple[int, float]]:
        q = expand_query(q)
        if method == "bm25":
            return self.sparse(q)[:k]
        if method == "dense":
            return self.dense(q)[:k]
        fused = self.rrf(self.dense(q), self.sparse(q))
        if method == "hybrid":
            return fused[:k]
        return self.rerank(q, fused[:CANDIDATES], k)


# ----------------------------------------------------------------------------------------
# Guardrails
# ----------------------------------------------------------------------------------------
BLOCK_PATTERNS = {
    "safety_bypass": r"\b(bypass|defeat|disable|override|jumper|tamper)\w*\b.{0,40}\b(interlock|alarm|safety|trip|"
                     r"sensor|relief|rupture|explosion vent|guard|e-?stop|emergency stop)",
    "weapons_or_harm": r"\b(make|build|cause|trigger)\b.{0,30}\b(explosion|bomb|explosive|fire)\b.{0,30}\b(on purpose|"
                       r"deliberately|intentionally|to (hurt|harm|kill|damage))|\b(hurt|harm|kill|injure)\b.{0,20}\b(someone|"
                       r"people|person|operator|worker)",
    "credentials_or_pii": r"\b(password|passcode|credential|api key|login|social security|ssn|home address|phone number)s?\b",
    "prompt_injection": r"ignore (all |the )?(previous|prior|above) (instructions|rules)|system prompt|you are now|"
                        r"jailbreak|developer mode",
}
SAFETY_CAUTION = r"\b(enter|entering|confined space|inside the cyclone|while (running|operating|in operation)|hot surface|" \
                 r"lockout|lock-out|isolation|explosive atmosphere|atex|zone 2[0-2]|maintenance on)\b"


def input_guardrail(q: str) -> tuple[bool, list[str]]:
    flags = []
    ql = q.lower()
    for name, pat in BLOCK_PATTERNS.items():
        if re.search(pat, ql):
            flags.append(f"blocked:{name}")
    if len(q.strip()) < 4:
        flags.append("blocked:empty_query")
    if re.search(SAFETY_CAUTION, ql):
        flags.append("caution:safety_critical_topic")
    return any(f.startswith("blocked") for f in flags), flags


BLOCK_MESSAGE = ("This request can't be answered by the documentation assistant (it asks to bypass safety functions, "
                 "cause harm, reveal credentials/personal data, or override the assistant's rules). For safety-related "
                 "changes contact the plant safety officer / follow the site Management-of-Change procedure.")
NO_ANSWER_MESSAGE = ("I could not find this in the indexed documentation, so I won't guess. Try rephrasing with equipment "
                     "terms (e.g. 'pressure drop', 'hopper', 'inlet velocity'), or consult the OEM / process engineer.")
RELATED_MESSAGE = ("The documentation does not answer this directly, so no answer is generated. The closest related "
                   "passages are quoted below for an engineer to interpret:")
SAFETY_NOTE = ("Safety note: follow site lock-out/tag-out, permit-to-work and the OEM manual; this assistant only "
               "summarises documentation and is not a substitute for a qualified person.")


# ----------------------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------------------
_LLM = {}

SYSTEM_PROMPT = ("You are a technical documentation assistant for industrial cyclone separators. Answer ONLY using the "
                 "numbered context passages. Cite every statement with the passage number in square brackets, e.g. [1]. "
                 "If the passages do not contain the answer, reply exactly: NOT_FOUND. Be concise (2-5 sentences). "
                 "Do not use outside knowledge.\n"
                 "Example format: Cyclone efficiency falls when the dust outlet plugs [2]. Inspect the hopper "
                 "regularly [1].")


def load_llm(kind: str):
    if kind in _LLM:
        return _LLM[kind]
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    torch.manual_seed(0)
    name = LLM_MODELS[kind]
    tok = AutoTokenizer.from_pretrained(name)
    if kind == "flan-t5":
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    _LLM[kind] = (tok, model)
    return _LLM[kind]


def build_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i + 1}] (source: {c.doc}, page {c.page})\n{c.text}" for i, c in enumerate(chunks))


def generate(kind: str, question: str, chunks: list[Chunk], max_new_tokens: int = 220) -> str:
    import torch
    tok, model = load_llm(kind)
    ctx = build_context(chunks)
    if kind == "flan-t5":
        prompt = (f"{SYSTEM_PROMPT}\n\nContext:\n{ctx}\n\nQuestion: {question}\nAnswer with citations:")
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, num_beams=2, do_sample=False)
        return tok.decode(out[0], skip_special_tokens=True).strip()
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context passages:\n{ctx}\n\nQuestion: {question}"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def extractive_answer(question: str, chunks: list[Chunk], n: int = 3) -> str:
    """Fallback 'generator': the most query-relevant sentences, quoted verbatim with citations."""
    cands = []
    for ci, c in enumerate(chunks):
        for s in split_sentences(c.text):
            if 6 <= len(s.split()) <= 70:
                cands.append((ci, s))
    if not cands:
        return NO_ANSWER_MESSAGE
    sc = reranker().predict([(question, s) for _, s in cands], show_progress_bar=False)
    best, seen = [], set()
    for i in np.argsort(-sc):                      # de-duplicate (overlapping chunks repeat sentences)
        key = re.sub(r"\W+", " ", cands[i][1].lower()).strip()
        if key in seen:
            continue
        seen.add(key); best.append(i)
        if len(best) == n:
            break
    best = sorted(best, key=lambda i: (cands[i][0], i))
    return " ".join(f"\"{cands[i][1]}\" [{cands[i][0] + 1}]" for i in best)


_NLI = None
_WINDOWS: dict[str, list[str]] = {}


def nli_model():
    global _NLI
    if _NLI is None:
        from sentence_transformers import CrossEncoder
        _NLI = CrossEncoder(NLI_MODEL, device="cpu")
        labels = {v.lower(): int(k) for k, v in _NLI.model.config.id2label.items()}
        _NLI.entail_idx = labels["entailment"]
    return _NLI


def source_windows(c: Chunk, width: int = 3) -> list[str]:
    """Premises for NLI: single sentences + overlapping 3-sentence windows (NLI models are trained on
    short premises; windows cover facts that span two sentences)."""
    if c.chunk_id not in _WINDOWS:
        ss = [x for x in split_sentences(c.text) if len(x.split()) >= 3] or [c.text]
        wins = [" ".join(ss[i:i + width]) for i in range(max(1, len(ss) - width + 1))] if len(ss) > 1 else []
        _WINDOWS[c.chunk_id] = ss + wins
    return _WINDOWS[c.chunk_id]


def entailment_by_chunk(sentence: str, chunks: list[Chunk]) -> np.ndarray:
    """max P(entailment) of `sentence` over the windows of each chunk -> array (n_chunks,)."""
    pairs, owner = [], []
    for ci, c in enumerate(chunks):
        for w in source_windows(c):
            pairs.append((w, sentence)); owner.append(ci)
    m = nli_model()
    logits = np.asarray(m.predict(pairs, show_progress_bar=False, batch_size=32))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = (p / p.sum(axis=1, keepdims=True))[:, m.entail_idx]
    out = np.zeros(len(chunks))
    for ci, v in zip(owner, p):
        out[ci] = max(out[ci], v)
    return out


def verify_and_cite(answer: str, chunks: list[Chunk]) -> tuple[list[dict], dict]:
    """Sentence-level faithfulness + citation enforcement with an NLI model.
    For every answer sentence: keep the LLM's citation if that passage entails the sentence; otherwise
    re-attribute to the passage that does (post-hoc attribution); if none does -> unsupported."""
    sents = [x for x in split_sentences(re.sub(r"\s+", " ", answer)) if len(re.sub(r"\[\d+\]", "", x).split()) >= 4]
    det, stats = [], {"kept": 0, "attributed": 0, "corrected": 0, "unsupported": 0}
    for x in sents:
        cited = [int(n) - 1 for n in re.findall(r"\[(\d+)\]", x) if 0 < int(n) <= len(chunks)]
        clean = re.sub(r"\s*\[\d+\]", "", x).strip()
        sc = entailment_by_chunk(clean.strip('"'), chunks)
        best = int(np.argmax(sc))
        if cited and max(sc[c] for c in cited) >= ENTAIL_MIN:
            refs, how = [c for c in cited if sc[c] >= ENTAIL_MIN], "kept"
        elif sc[best] >= ENTAIL_MIN:
            refs, how = [best], ("corrected" if cited else "attributed")
        else:
            refs, how = [], "unsupported"
        stats[how] += 1
        det.append({"sentence": clean, "cited": [r + 1 for r in refs], "entailment": round(float(sc[best]), 3),
                    "supported": bool(refs), "how": how})
    return det, stats


def render(det: list[dict]) -> str:
    out = []
    for d in det:
        s = re.sub(r"\*\*|^\s*\d+\.\s+|^[-•]\s+", "", d["sentence"]).strip()
        if s.endswith(":"):
            continue                      # dangling list header whose items were removed / restructured
        tail = "".join(f"[{c}]" for c in d["cited"])
        out.append(f"{s} {tail}".strip())
    return " ".join(out)


# ----------------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------------
class RAG:
    def __init__(self, llm: str = "qwen", index_dir: Path = INDEX_DIR):
        self.retriever = Retriever(index_dir)
        self.llm = llm

    def ask(self, question: str, k: int = TOP_K) -> Answer:
        t = {}
        t0 = time.time()
        blocked, flags = input_guardrail(question)
        if blocked:
            return Answer(question, BLOCK_MESSAGE, "blocked", guardrails=flags)

        hits = self.retriever.search(question, "hybrid+rerank", k)
        t["retrieval"] = round(time.time() - t0, 2)
        retrieved = [{"rank": r + 1, "doc": self.retriever.chunks[i].doc, "page": self.retriever.chunks[i].page,
                      "rerank_score": round(s, 3), "chunk_id": self.retriever.chunks[i].chunk_id}
                     for r, (i, s) in enumerate(hits)]
        if expand_query(question) != question:
            flags.append("info:query_expanded_with_glossary")
        if not hits or hits[0][1] < RELATED_MIN_SCORE:
            flags.append("fallback:no_relevant_passage")
            return Answer(question, NO_ANSWER_MESSAGE, "no_answer", guardrails=flags, retrieved=retrieved, timings_s=t)
        if hits[0][1] < RERANK_MIN_SCORE:
            # weakly related material only: do not generate; show the closest passages verbatim so a human can judge
            flags.append("fallback:only_related_passages")
            chunks = [self.retriever.chunks[i] for i, s in hits[:3] if s >= RELATED_MIN_SCORE]
            quotes = extractive_answer(question, chunks, n=2)
            cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", quotes)})
            cites = [{"ref": n, "doc": chunks[n - 1].doc, "page": chunks[n - 1].page,
                      "snippet": textwrap.shorten(chunks[n - 1].text, 260)} for n in cited]
            msg = (f"{RELATED_MESSAGE}\n{quotes}")
            return Answer(question, msg, "related_only", citations=cites, guardrails=flags, retrieved=retrieved, timings_s=t)

        # keep only passages that individually clear the relevance gate
        kept = [(i, s) for i, s in hits if s >= RERANK_MIN_SCORE]
        chunks = [self.retriever.chunks[i] for i, _ in kept]

        t1 = time.time()
        if self.llm == "extractive":
            raw = extractive_answer(question, chunks)
        else:
            raw = generate(self.llm, question, chunks)
        t["generation"] = round(time.time() - t1, 2)

        answer, status = raw, "answered"
        if "NOT_FOUND" in raw.upper().replace(" ", "_") or not raw.strip():
            flags.append("fallback:llm_reported_not_found")
            return Answer(question, NO_ANSWER_MESSAGE, "no_answer", guardrails=flags, retrieved=retrieved, timings_s=t)

        # Citation enforcement + faithfulness (NLI): each sentence must be entailed by the passage it cites.
        # Unsupported sentences are removed; if fewer than half survive, fall back to quoting the sources.
        det, st = verify_and_cite(answer, chunks)
        frac = float(np.mean([d["supported"] for d in det])) if det else 0.0
        if st["attributed"] or st["corrected"]:
            flags.append(f"guardrail:citations_attributed={st['attributed']},corrected={st['corrected']}")
        if frac < 0.5:
            flags.append(f"guardrail:low_faithfulness({frac:.2f})->extractive")
            answer = extractive_answer(question, chunks)
            det, _ = verify_and_cite(answer, chunks)
            status = "low_confidence" if not any(d["supported"] for d in det) else "answered_extractive"
        else:
            if st["unsupported"]:
                flags.append(f"guardrail:removed_{st['unsupported']}_unsupported_sentence(s)")
            answer = render([d for d in det if d["supported"]])
        flags.append(f"info:supported_sentence_fraction={frac:.2f}")
        if any(f.startswith("caution") for f in flags):
            answer = f"{answer}\n\n{SAFETY_NOTE}"

        cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer) if 0 < int(x) <= len(chunks)})
        cites = [{"ref": n, "doc": chunks[n - 1].doc, "page": chunks[n - 1].page,
                  "snippet": textwrap.shorten(chunks[n - 1].text, 260)} for n in cited]
        t["total"] = round(time.time() - t0, 2)
        return Answer(question, answer, status, citations=cites, guardrails=flags, retrieved=retrieved, timings_s=t)


def print_answer(a: Answer) -> None:
    print("\n" + "=" * 100)
    print(f"Q: {a.question}\n")
    print(textwrap.fill(a.answer, 100, replace_whitespace=False) if "\n" not in a.answer else a.answer)
    print(f"\nstatus: {a.status}   guardrails: {', '.join(a.guardrails) or 'none'}   timings: {a.timings_s}")
    if a.citations:
        print("\nSources:")
        for c in a.citations:
            print(f"  [{c['ref']}] {c['doc']}, p.{c['page']}: \"{c['snippet']}\"")
    elif a.retrieved and a.status == "no_answer":
        print("\nClosest passages (below relevance threshold):")
        for r in a.retrieved[:3]:
            print(f"  - {r['doc']}, p.{r['page']} (score {r['rerank_score']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="build the index from ./docs")
    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--llm", choices=["qwen", "flan-t5", "extractive"], default="qwen")
    a.add_argument("--json", action="store_true", help="print the full JSON result")
    c = sub.add_parser("chat", help="interactive Q&A loop")
    c.add_argument("--llm", choices=["qwen", "flan-t5", "extractive"], default="qwen")
    args = ap.parse_args()

    if args.cmd == "ingest":
        ingest()
    elif args.cmd == "ask":
        res = RAG(args.llm).ask(args.question)
        print(json.dumps(asdict(res), indent=2)) if args.json else print_answer(res)
    else:
        rag = RAG(args.llm)
        print("Cyclone documentation assistant - type a question (empty line to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            print_answer(rag.ask(q))


if __name__ == "__main__":
    main()
