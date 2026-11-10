# Data Science take-home: Prajot Shrivastava

There are two parts:
1. **Task 1:** analysis of 3.6 years of 5-minute cyclone sensor data (shutdowns, operating states, contextual anomalies, 1-hour forecasting, insights).
2. **Task 2:** a RAG system design plus a runnable open-source prototype that answers questions over the 11 cyclone documents from the provided Google Drive folder.

The submission folder is named `Prajot_Shrivastava_DataScience`. The name comes from the git author identity configured on the machine where this was built.

## Contents

```
Prajot_Shrivastava_DataScience/
├── README.md                     ← this file
├── Final_Presentation.pptx       ← 8 slides covering both tasks
├── Task1/
│   ├── README.md                 ← method, results, case-study hypotheses, insights, run steps
│   ├── task1_analysis.py         ← end-to-end runnable pipeline
│   ├── requirements.txt
│   ├── data/data.xlsx            ← copy of the provided dataset
│   ├── shutdown_periods.csv      ← required outputs (4 CSVs)
│   ├── anomalous_periods.csv
│   ├── clusters_summary.csv
│   ├── forecasts.csv
│   ├── plots/                    ← 22 PNG figures
│   └── outputs/                  ← supporting outputs (data-quality report, metrics, labels, run log, summary JSON)
├── Task2/
│   ├── architecture_diagram.pptx ← architecture (slide 1) + deployment/scaling view (slide 2), editable shapes
│   ├── notes.md                  ← design trade-offs, retrieval strategy, guardrails, metrics, scaling plan
│   └── prototype/
│       ├── README.md             ← exact setup, model downloads, run and evaluation commands
│       ├── rag_prototype.py      ← ingest / ask / chat
│       ├── evaluate.py, eval_questions.json
│       ├── requirements.txt
│       ├── docs/                 ← the 11 corpus PDFs (from the assignment's Drive folder)
│       ├── index/                ← prebuilt FAISS + BM25 index
│       ├── evaluation.csv, evaluation_summary.csv, guardrail_eval.csv,
│       │   end_to_end_eval.csv, end_to_end_summary.csv, eval_answers.jsonl
│       └── demo_output.txt       ← example transcript
└── presentation/make_slides.py   ← regenerates both .pptx files from the result files
```

## Reproduce everything

Requirements: Python 3.10+ (developed and tested on **3.14, macOS arm64, CPU only**), about 3 GB RAM, about 2 GB disk for models, and internet on the first run of Task 2 for Hugging Face model downloads.

```bash
cd Prajot_Shrivastava_DataScience
python3 -m venv .venv && source .venv/bin/activate
pip install -r Task1/requirements.txt -r Task2/prototype/requirements.txt python-pptx==1.0.2

# Task 1 (~1.5 min): regenerates the 4 CSVs, plots/ and outputs/
python Task1/task1_analysis.py

# Task 2: index is prebuilt; rebuild + ask + evaluate
cd Task2/prototype
python rag_prototype.py ingest
python rag_prototype.py ask "What are the primary indicators of cyclone performance?"
python evaluate.py --llm qwen          # retrieval + guardrail + end-to-end metrics (~2 min after downloads)
cd ../..

# Slides (read numbers from the files above)
python presentation/make_slides.py
```

## Headline results

**Task 1** (details in `Task1/README.md`)
- **Data cleaning:**
  - status strings such as "Not Connect" and "I/O Timeout" (8,195 cells) → NaN
  - 3 missing days (every 31 Dec) re-gridded
  - saturated, stuck and physically implausible readings → NaN
  - outliers kept for the anomaly detector
- **Shutdowns:** 154 events and 6,840 h of downtime (21.7 % of the record), detected by draft collapse (ID fan off) or inlet gas < 300 °C. 33 stops longer than 24 h hold about 95 % of downtime.
- **Operating states:** 4 (KMeans on 1-hour regime features): High load, Normal, Reduced draft, Unstable. Their mix drifts strongly between 2017 and 2020.
- **Anomalies:** 746 state-contextual events (per-state Isolation Forest + robust z), with 5 evidence-based root-cause hypotheses. The most severe events are **instrument faults**, including a months-long outlet-thermocouple problem in 2018. Anomalies are about 12× more frequent just before stops.
- **Forecasting:** a 1-hour forecast of inlet gas temperature with HistGradientBoosting has RMSE 15.0 / MAE 9.2 °C, against 19.3 / 11.8 for persistence, on a time-ordered 2020 hold-out.

**Task 2** (details in `Task2/notes.md`)
- **Pipeline:** hybrid BM25 + bge-small retrieval, cross-encoder rerank and relevance gate, Qwen2.5-0.5B-Instruct generation, and an NLI check of every answer sentence against the passage it cites. Unsupported sentences are removed, with a fallback to quotes.
- **Retrieval:** passage hit@3 of 0.94 on 17 labelled questions.
- **Guardrails:** 4/4 out-of-scope questions refused, 4/4 sensitive prompts blocked, and no answerable questions refused.
- **Answers:** every answer is cited, with a median of about 3.6 s on CPU.

## Honest scope notes

- **Corpus:** the Google Drive corpus *was* accessible, so the RAG uses the real 11 PDFs, not synthetic documents. 18 image-only pages are not indexed (no OCR).
- **Evaluation:** the RAG evaluation set is small and labelled by me, and its thresholds were set by looking at that data. Treat the metrics as smoke tests.
- **Validation:** Task 1 root causes are **hypotheses from sensor data only**. No operator or maintenance logs were available to confirm shutdown causes or anomalies.
- **Development environment:** I worked in a virtualenv outside this folder. The commands above create a fresh one. Hugging Face model files are cached in the user's home directory, not in this folder.
