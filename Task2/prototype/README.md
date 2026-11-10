# RAG prototype: cyclone documentation Q&A with citations

This is a minimal but complete retrieval-augmented QA system over the **11 real PDFs from the assignment's Google Drive folder** (in `docs/`). It uses only free, open-source models and libraries, runs on a laptop CPU, and was executed end to end while building it. See `demo_output.txt` and the evaluation files.

```
question ─► input guardrail ─► hybrid retrieval (bge-small dense + BM25, RRF) ─► cross-encoder rerank
        ─► relevance gate ─► Qwen2.5-0.5B-Instruct (grounded prompt) ─► NLI faithfulness + citation check ─► answer [n] + sources
```

## Files

| File | Purpose |
|---|---|
| `rag_prototype.py` | The whole system: `ingest` (PDF → chunks → FAISS + BM25), `ask`, `chat` |
| `evaluate.py` | Retrieval metrics (precision@k, recall@k, hit@k, MRR), guardrail checks, optional end-to-end answer evaluation |
| `eval_questions.json` | 17 hand-labelled answerable questions, 4 out-of-scope and 4 sensitive prompts |
| `docs/` | The 11 corpus PDFs (downloaded from the Drive folder, byte sizes verified) |
| `index/` | Built index (`dense.faiss`, `bm25.pkl`, `chunks.jsonl`, `manifest.json`). Rebuild it with `ingest`. |
| `evaluation.csv` / `evaluation_summary.csv` | Retrieval metrics per question × retriever × k, and their means |
| `guardrail_eval.csv` | Gate decision for every test prompt (passes / no_answer / related_only / blocked) |
| `end_to_end_eval.csv`, `end_to_end_summary.csv`, `eval_answers.jsonl` | Full-pipeline answers and metrics with Qwen2.5-0.5B |
| `demo_output.txt` | Example transcript: answered, related-only, extractive fallback, refused and blocked cases |

## Setup (exact steps)

Tested on macOS arm64 with Python 3.14, CPU only. Linux and Windows should work the same way, since nothing here is Mac-specific apart from the import-order note below.

```bash
cd Task2/prototype
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Models (downloaded automatically from Hugging Face on first use, ~1.5 GB in total)

| Role | Model | Licence | Download | When needed |
|---|---|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` | MIT | ~135 MB | ingest + every query |
| Reranker / relevance gate | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 | ~90 MB | every query |
| Faithfulness (NLI) | `cross-encoder/nli-deberta-v3-xsmall` | Apache-2.0 | ~280 MB | every generated answer |
| Generator (default) | `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 | ~1.0 GB | `--llm qwen` |
| Generator (alternative) | `google/flan-t5-base` | Apache-2.0 | ~1.0 GB | `--llm flan-t5` (wired in but **not evaluated**) |

To pre-download everything (for example before going offline):

```bash
python -c "from huggingface_hub import snapshot_download as d; [d(m) for m in ['BAAI/bge-small-en-v1.5','cross-encoder/ms-marco-MiniLM-L-6-v2','cross-encoder/nli-deberta-v3-xsmall','Qwen/Qwen2.5-0.5B-Instruct']]"
```

After that you can run with `HF_HUB_OFFLINE=1`. Expected resources: about 3 GB RAM, no GPU. Building the index takes about 12 s. With models already loaded, a query takes about 0.2 s for retrieval and a median of about 3.6 s end to end with Qwen on an Apple-silicon CPU. A fresh `ask` process also spends a few seconds loading models.

## Run

```bash
python rag_prototype.py ingest                                   # builds index/ from docs/ (already built in the submission)
python rag_prototype.py ask "What are the primary indicators of cyclone performance?"
python rag_prototype.py ask "What is roping in a cyclone?" --llm extractive   # no generator download: quotes sources
python rag_prototype.py ask "..." --json                         # full result: answer, status, citations, flags, timings
python rag_prototype.py chat                                     # interactive loop
python evaluate.py                                               # retrieval + guardrail metrics (fast, no LLM)
python evaluate.py --llm qwen                                    # + end-to-end answers (~2 min)
```

**Adding the full or real document set:** drop more `.pdf`, `.txt` or `.md` files into `docs/` and rerun `ingest`. Image-only (scanned) pages are reported during ingest. They need OCR, which is not included.

## Output statuses

| status | meaning |
|---|---|
| `answered` | LLM answer. Every sentence was verified by NLI against the passage it cites; unsupported sentences were removed. |
| `answered_extractive` | The LLM draft was mostly unsupported (< 50 % of sentences entailed), so verbatim quotes with citations are returned instead |
| `related_only` | Only weakly related passages were found (rerank score between −6 and −4). They are quoted and no answer is generated. |
| `no_answer` | Nothing relevant was found (or the LLM said NOT_FOUND), so the system returns a fallback message instead of guessing |
| `blocked` | The input policy matched (safety bypass, harm, credentials/PII, prompt injection) |

## Evaluation results (from the files in this folder)

Retrieval, mean over 17 questions (`evaluation_summary.csv`, k = 3):

| Retriever | precision@3 (doc) | recall@3 (doc) | hit@3 (passage) | MRR@10 |
|---|---|---|---|---|
| BM25 | 0.78 | 0.92 | 0.88 | 0.87 |
| Dense | 0.71 | 0.97 | 0.88 | 0.79 |
| Hybrid RRF | 0.75 | 0.97 | 0.94 | 0.88 |
| Hybrid + rerank (used) | 0.77 | 0.98 | 0.94 | 0.85 |

End-to-end results with Qwen2.5-0.5B are in `end_to_end_summary.csv`:
- **Guardrails:** 4/4 out-of-scope questions refused, 4/4 sensitive prompts blocked, 0/17 answerable questions refused.
- **Answers:** 100 % carry citations, and about 94 % cite a relevant document. About 76 % are kept as LLM answers; the rest fall back to quotes.
- **Faithfulness:** only about 73 % of the raw LLM draft sentences were entailed by their sources. That is why the NLI filter exists.

**Limitations:** the evaluation set is small and labelled by me. The thresholds were set by looking at this data, so they need recalibrating on a larger held-out set. The regex policy is not a substitute for a trained safety classifier. Tables are flattened, and scanned pages are not indexed. See `../notes.md` for the full discussion.

## Troubleshooting

- **The process exits silently while embedding (macOS):** `faiss` was imported before `torch`. The code always imports torch first (`import_faiss()`); keep that order if you modify it.
- **`DependencyError: cryptography>=3.1 is required`:** run `pip install cryptography`. One corpus PDF is AES-encrypted with an owner password.
- **Slow first query:** model downloads and loading. Later queries in `chat` are fast.
