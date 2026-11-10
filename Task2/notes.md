# Task 2: RAG for cyclone technical documentation (design notes)

This document covers the design trade-offs, retrieval strategy, guardrails, evaluation and scaling plan. The runnable prototype is in [`prototype/`](prototype/README.md), and the architecture diagram is in [`architecture_diagram.pptx`](architecture_diagram.pptx).

## 0. Corpus actually used

The Google Drive folder linked in the assignment ("Task 2 Doc") was accessible. All **11 PDFs** were downloaded, and each file's byte size was checked against the Drive metadata. They are in `prototype/docs/` and are what the prototype indexes and is evaluated on. Nothing synthetic was added.

| Property | Value |
|---|---|
| Documents / pages / chunks | 11 / 215 / 355 |
| Pages with no extractable text (image-only) | 18. The text on these pages is **not indexed**. OCR is listed as future work. |
| Special cases | `172-65231ma.pdf` is AES-encrypted with only an owner password; pypdf opens it with an empty password, which needs the `cryptography` package. `SOP Cyclone-22022023.pdf` is a maritime **cyclone-storm** SOP, not a separator manual. It shows real term ambiguity: questions about "cyclone warnings" must retrieve it, and separator questions must not. |

## 1. Architecture

```
OFFLINE  sources ─► parse & clean ─► chunk (page-aware) ─► embed ─► vector index + BM25 + metadata
ONLINE   question ─► input guardrail ─► hybrid retrieval (dense+BM25, RRF) ─► cross-encoder rerank
         ─► relevance gate ─► grounded LLM ─► NLI faithfulness + citation enforcement ─► answer + sources
                     │blocked                 │below τ: "not in docs"        │<50 % supported: verbatim quotes
MONITOR  golden-set metrics (P@k, R@k, MRR, faithfulness, refusal accuracy) + online telemetry
```

| Layer | Prototype (runs on a laptop CPU) | Production choice | Why |
|---|---|---|---|
| Ingestion | `pypdf` text per page; removes headers/footers repeated on ≥ 50 % of pages, table-of-contents leader lines and bare page numbers; de-hyphenation | Same, plus a layout-aware parser (e.g. Docling or Unstructured) for tables, OCR (Tesseract/PaddleOCR) for image-only pages, a document registry (hash, version, OEM, equipment, doc type, language), and incremental updates | Manuals are table- and figure-heavy. Troubleshooting tables lose their row structure as plain text. |
| Chunking | Sentence-aware sliding window, **~180 words (~240 tokens), 40-word overlap**, never crosses a page boundary, minimum 25 words | Structure-aware (section headings), troubleshooting-table rows kept as atomic chunks, parent-section expansion for the LLM | Fits the 512-token limit of the embedder and reranker with headroom. Page-bounded chunks make every citation a precise doc + page. The overlap keeps facts that straddle a boundary. |
| Embeddings | `BAAI/bge-small-en-v1.5` (33 M parameters, 384 dimensions, MIT licence) with the BGE query instruction prefix | `bge-base-en-v1.5`, or `bge-m3` if documents are multilingual (the SOP is bilingual Hindi/English) | Strong retrieval quality for its size and fast on CPU. Ingest took 12 s for 355 chunks. |
| Index | FAISS `IndexFlatIP` (exact cosine) + `rank_bm25` | Qdrant / pgvector (HNSW) with payload filters; BM25 via OpenSearch or Qdrant sparse vectors | Exact search is fine below about 1 M vectors. A database is needed for metadata filters, updates and access control. |
| Retrieval | Dense top-20 + BM25 top-20 → reciprocal-rank fusion (k = 60) → `cross-encoder/ms-marco-MiniLM-L-6-v2` rerank → top 4 | Same, plus query rewriting (abbreviations like "ID fan", units) and metadata filters | BM25 catches exact part numbers, units and codes; dense search catches paraphrases. RRF needs no score calibration. |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, CPU, greedy decoding). Alternatives: `google/flan-t5-base`, or `--llm extractive` (no generator) | 7–8B instruct model (Qwen2.5-7B / Llama-3.1-8B), AWQ int4, served by vLLM | 0.5B keeps the prototype runnable anywhere. The guardrails below matter more at this size. |

## 2. Retrieval strategy and measured quality

There are 17 hand-labelled answerable questions in `prototype/eval_questions.json`. I labelled them by reading the PDFs. A passage counts as "relevant" if it comes from a relevant document **and** contains a key phrase. Results are in `prototype/evaluation_summary.csv`:

| Retriever | P@1 (doc) | Hit@3 (passage) | R@3 (doc) | MRR@10 (passage) |
|---|---|---|---|---|
| BM25 | 0.94 | 0.88 | 0.92 | 0.87 |
| Dense (bge-small) | 0.82 | 0.88 | 0.97 | 0.79 |
| Hybrid (RRF) | 0.94 | **0.94** | 0.97 | **0.88** |
| Hybrid + rerank (used) | 0.94 | **0.94** | **0.98** | 0.85 |

Reading the results honestly:
- **Hybrid beats either retriever alone.** Dense search misses exact tokens such as "30 psig", and BM25 misses paraphrases.
- **The cross-encoder did not improve the ranking on this set.** The questions were written by someone who had just read the documents, so they reuse document wording, which favours lexical matching. I still keep the reranker because its score is a calibrated relevance signal: it is what powers the "not in the documentation" gate. That gate cleanly separated answerable questions (minimum score −0.33) from out-of-scope ones (maximum −9.8).

**Faithfulness and relevance mechanisms:** answers must cite passages, each citation shows doc, page and a snippet, and each sentence is checked by NLI (section 3).

## 3. Guardrails and failure modes

| Failure mode | Mechanism (prototype) | Evidence |
|---|---|---|
| **No relevant answer** | Rerank score of the top passage < τ = −4.0 → fixed fallback message plus the nearest sources, with **no LLM call**. If the LLM itself replies `NOT_FOUND`, the same fallback is used. | 4/4 out-of-scope questions refused; 17/17 answerable passed (`guardrail_eval.csv`). τ was set in the middle of the observed score gap, so it should be recalibrated on a larger held-out set. |
| **Hallucination / unsupported claims** | (1) Grounded prompt: answer only from numbered passages, cite `[n]`, or say NOT_FOUND. (2) Citations to non-existent passages are dropped. (3) **Sentence-level NLI check** (`cross-encoder/nli-deberta-v3-xsmall`): each sentence must be entailed (P ≥ 0.1) by a single sentence or a 3-sentence window of the passage it cites. Otherwise it is re-attributed to the passage that does entail it, or marked unsupported. (4) Unsupported sentences are removed. If fewer than half survive, the answer is replaced by **verbatim quotes** from the top passages, with citations. | Only 73 % of the 0.5B model's draft sentences were entailed. The filter removed a fabricated mechanism ("increasing inertial forces…") and a wrong fact (the TLV separator removing "dust particles"; the PDF says condensate). After filtering: 100 % of answers cite sources, 94 % cite a relevant document, 24 % fell back to quotes. |
| **Sensitive queries** | Input policy (regex) blocks safety-function bypass (interlocks, alarms, trips, relief or explosion vents), intent to harm, credentials/PII and prompt-injection phrases, and replies with a refusal that points to the safety officer / management-of-change process. Safety-critical but legitimate topics (entering the cyclone, ATEX zones, lock-out) are answered with an extra safety note. | 4/4 sensitive test prompts blocked. These are hand-written rules tested on hand-written prompts, so they are indicative only. Production should add a learned classifier (e.g. Llama Guard 3 1B or Prompt-Guard) and red-team prompts. |
| Prompt injection inside documents | Passages go into the prompt as quoted, numbered data, and the system prompt forbids following instructions found in them | Not tested. Production should scan ingested text for instruction-like content. |
| Over-refusal | Measured as the false-refusal rate on answerable questions | 0/17 at the gate. NLI strictness, not refusals, causes the 24 % fallback-to-quotes. |

The **threshold choices** (τ = −4.0, NLI P ≥ 0.1) were tuned on small samples I looked at myself (the evaluation set, and 12 labelled answer sentences where unsupported sentences scored ≤ 0.027). They are reasonable starting points, not validated operating points.

### Monitoring metrics

- **Offline** (re-run on every index, model or prompt change; `prototype/evaluate.py`):
  - precision@k and recall@k (document level), passage hit@k and MRR
  - share of answer sentences entailed by their citations (NLI faithfulness)
  - citation correctness (cites a relevant doc)
  - key-fact coverage
  - refusal accuracy on out-of-scope, false-refusal rate on in-scope, block rate on sensitive prompts
- **Online:**
  - share of queries hitting each fallback (no-answer, quote fallback, blocked)
  - distribution of top rerank scores (drift alarm)
  - thumbs up/down and citation click-through
  - p50/p95 latency and cost per answer
  - clusters of unanswered queries → missing or poorly parsed documents

## 4. Scalability and cost

| Concern | Plan |
|---|---|
| **10× documents** (~110 PDFs → ~3.5 k chunks; even 500 PDFs ≈ 20–50 k chunks) | Still small for any vector database. Ingestion becomes an incremental batch job keyed on document hash, so changed files are upserted without a full rebuild. Use HNSW (Qdrant/pgvector) with metadata filters (plant, equipment, doc type) to keep precision as the corpus diversifies. Re-embedding 50 k chunks with bge-small on CPU takes minutes (measured: 355 chunks in ~12 s including load), or seconds on a spot GPU. OCR only image-only pages. |
| **100+ concurrent users** | Retrieval (query embedding, BM25, rerank) takes about 0.2 s of CPU and is stateless, so pods scale horizontally. The LLM is the bottleneck, so serve one shared, quantised 7–8B model with **vLLM continuous batching** and autoscale on queue depth. Users read answers between questions, so 100 concurrent users is likely only a few requests per second at peak, which one or two 24 GB GPU replicas should handle. **This must be confirmed by a load test.** Add a semantic cache (Redis) for repeated questions, streaming responses, and per-user rate limits. |
| **Cost under constraints** | Run ingestion and retrieval serverless or scaled low. Size the GPU pool for the peak shift and shrink it off-shift; use spot instances for re-indexing. Route simple look-ups to the extractive path or a small model, and use the 7–8B model only when needed. Quantise to int4 (AWQ) so it fits on a single mid-range GPU. Track cost per answered question next to quality. |

## 5. Trade-offs and limitations

- A **0.5B generator** keeps the prototype laptop-runnable (median 3.6 s per answer on CPU). Its drafts are often imprecise, which is why NLI filtering and quote fallback exist. A 7–8B model would raise the share of answers kept as-is.
- **NLI-xsmall is strict**. It rejects some correct paraphrases, for example naming "DC7" when the passage says "the separator". That costs fluency, not correctness. Larger NLI models were tried (deberta-v3-small and -base) and did not separate better on the labelled sample.
- 18 image-only pages are not indexed (no OCR in the prototype), and tables are flattened to text.
- The evaluation set is small (17 + 4 + 4) and author-labelled. Treat the numbers as smoke tests, not benchmarks.
