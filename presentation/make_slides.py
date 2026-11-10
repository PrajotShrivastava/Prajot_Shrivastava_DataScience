#!/usr/bin/env python3
"""
Builds  ../Final_Presentation.pptx  and  ../Task2/architecture_diagram.pptx  with python-pptx.
All numbers are read from the Task 1 / Task 2 output files, so the decks stay consistent with the analysis.

Run (after Task1/task1_analysis.py and Task2/prototype/evaluate.py --llm qwen):
    python presentation/make_slides.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
T1, T2 = ROOT / "Task1", ROOT / "Task2"
PLOTS = T1 / "plots"

NAVY = RGBColor(0x14, 0x2B, 0x45)
TEAL = RGBColor(0x0F, 0x7C, 0x8C)
ORANGE = RGBColor(0xE0, 0x7A, 0x1F)
RED = RGBColor(0xB8, 0x3B, 0x3B)
GREY = RGBColor(0x5F, 0x6B, 0x76)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1E, 0x24, 0x2B)

W, H = Inches(13.333), Inches(7.5)


def r2(x, nd=2) -> str:
    """Round half-up (matches the hand-written READMEs; avoids float '0.745' -> '0.74')."""
    from decimal import ROUND_HALF_UP, Decimal
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------------- helpers
def new_deck() -> Presentation:
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p


def blank(p):
    return p.slides.add_slide(p.slide_layouts[6])


def text(slide, x, y, w, h, s, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = s if isinstance(s, list) else [s]
    for i, ln in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        r = para.add_run()
        r.text = ln
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, "Calibri"
    return tb


def bullets(slide, x, y, w, h, items, size=14, gap=6):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for it in items:
        lvl = 1 if it.startswith("  ") else 0
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.space_after = Pt(gap)
        head, _, rest = it.strip().partition("|")
        r = para.add_run()
        r.text = ("•  " if lvl == 0 else "–  ") + head
        r.font.size = Pt(size - 2 * lvl)
        r.font.bold = bool(rest) and lvl == 0
        r.font.color.rgb = NAVY if (rest and lvl == 0) else INK
        r.font.name = "Calibri"
        if rest:
            r2 = para.add_run()
            r2.text = " " + rest
            r2.font.size, r2.font.color.rgb, r2.font.name = Pt(size - 2 * lvl), INK, "Calibri"
        if lvl:
            para.level = 1
    return tb


def header(slide, title, kicker=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.95))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    text(slide, Inches(0.45), Inches(0.12), Inches(10.5), Inches(0.75), title, 26, True, WHITE, anchor=MSO_ANCHOR.MIDDLE)
    if kicker:
        text(slide, Inches(9.6), Inches(0.12), Inches(3.4), Inches(0.75), kicker, 12, False,
             RGBColor(0xB9, 0xD7, 0xDC), PP_ALIGN.RIGHT, MSO_ANCHOR.MIDDLE)


def footer(slide, n):
    text(slide, Inches(0.45), Inches(7.08), Inches(9), Inches(0.3),
         "Prajot Shrivastava · Data Science take-home · Cyclone analytics + documentation RAG", 9, color=GREY)
    text(slide, Inches(12.2), Inches(7.08), Inches(0.7), Inches(0.3), str(n), 9, color=GREY, align=PP_ALIGN.RIGHT)


def box(slide, x, y, w, h, title, body="", fill=LIGHT, edge=TEAL, title_color=NAVY, size=11, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = edge; s.line.width = Pt(1.25)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.12
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p0 = tf.paragraphs[0]; p0.alignment = PP_ALIGN.CENTER
    r = p0.add_run(); r.text = title
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size + 1), True, title_color, "Calibri"
    if body:
        p1 = tf.add_paragraph(); p1.alignment = PP_ALIGN.CENTER
        r = p1.add_run(); r.text = body
        r.font.size, r.font.color.rgb, r.font.name = Pt(size - 1), INK, "Calibri"
    return s


def arrow(slide, x1, y1, x2, y2, color=GREY, width=1.75, dashed=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dashed:  # schema order inside <a:ln>: fill, prstDash, join, headEnd, tailEnd
        ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def right_mid(s):
    return s.left + s.width, s.top + s.height // 2


def left_mid(s):
    return s.left, s.top + s.height // 2


def bottom_mid(s):
    return s.left + s.width // 2, s.top + s.height


def top_mid(s):
    return s.left + s.width // 2, s.top


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(str(path), x, y, w, h)


def table(slide, x, y, w, df: pd.DataFrame, size=11, col_w=None, highlight_col=None):
    rows, cols = df.shape[0] + 1, df.shape[1]
    t = slide.shapes.add_table(rows, cols, x, y, w, Inches(0.3) * rows).table
    for j, c in enumerate(df.columns):
        cell = t.cell(0, j); cell.text = str(c)
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        pr = cell.text_frame.paragraphs[0]; pr.runs[0].font.size = Pt(size); pr.runs[0].font.bold = True
        pr.runs[0].font.color.rgb = WHITE
    for i in range(df.shape[0]):
        for j in range(cols):
            cell = t.cell(i + 1, j); cell.text = str(df.iat[i, j])
            cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT if i % 2 == 0 else WHITE
            r = cell.text_frame.paragraphs[0].runs[0]
            r.font.size = Pt(size); r.font.color.rgb = INK
            if highlight_col is not None and i == highlight_col:
                r.font.bold = True
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = cw
    return t


# ---------------------------------------------------------------------------------- architecture
def draw_architecture(slide, x0=Inches(0.35), y0=Inches(1.15), compact=False):
    """Two-lane RAG architecture drawn with native (editable) shapes."""
    bw, bh, gap = Inches(1.92), Inches(1.0), Inches(0.22)
    # lane labels
    for i, (lbl, col) in enumerate([("OFFLINE · ingestion & indexing", TEAL), ("ONLINE · query answering", ORANGE)]):
        y = y0 + i * Inches(2.05)
        lab = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x0, y, Inches(12.6), Inches(0.3))
        lab.fill.solid(); lab.fill.fore_color.rgb = col; lab.line.fill.background()
        tf = lab.text_frame; tf.margin_left = Inches(0.1)
        r = tf.paragraphs[0].add_run(); r.text = lbl
        r.font.size, r.font.bold, r.font.color.rgb = Pt(11), True, WHITE

    yA = y0 + Inches(0.45)
    a = [
        ("Document sources", "Drive / SharePoint / S3\nmanuals · SOPs · guides"),
        ("Parse & clean", "pypdf text, header/footer &\nTOC removal · OCR for scans"),
        ("Chunking", "sentence-aware, ~180 words\n40-word overlap · page meta"),
        ("Embeddings", "bge-small-en-v1.5 (384-d)\nbatch job · doc-hash dedupe"),
        ("Indexes", "Vector: FAISS → Qdrant/pgvector\nSparse: BM25 · metadata store"),
    ]
    A = []
    for i, (t, b) in enumerate(a):
        A.append(box(slide, x0 + i * (bw + gap) + Inches(0.15), yA, bw, bh, t, b, edge=TEAL, size=10))
    for s1, s2 in zip(A, A[1:]):
        arrow(slide, *right_mid(s1), *left_mid(s2))

    yB = y0 + Inches(2.5)
    b = [
        ("User question", "operator / engineer\nnatural language"),
        ("Input guardrail", "policy & safety-bypass block\nprompt-injection · PII"),
        ("Hybrid retrieval", "dense + BM25 top-20 each\nreciprocal-rank fusion"),
        ("Rerank + gate", "cross-encoder MiniLM\nscore < τ → 'not in docs'"),
        ("Grounded LLM", "Qwen2.5-0.5B (prototype)\n7-8B AWQ via vLLM (prod)"),
        ("Output guardrails", "NLI entailment per sentence\ncitations · safety note"),
    ]
    bw2 = Inches(1.92)
    gap2 = Inches(0.17)
    B = []
    for i, (t, bdy) in enumerate(b):
        B.append(box(slide, x0 + i * (bw2 + gap2) + Inches(0.05), yB, bw2, bh, t, bdy, edge=ORANGE, size=10))
    for s1, s2 in zip(B, B[1:]):
        arrow(slide, *right_mid(s1), *left_mid(s2))
    # indexes feed retrieval
    arrow(slide, *bottom_mid(A[4]), B[2].left + B[2].width // 2 + Inches(0.4), B[2].top, color=TEAL, dashed=True)

    # answer + fallbacks
    yC = yB + bh + Inches(0.35)
    ans = box(slide, B[5].left - Inches(0.2), yC, bw2 + Inches(0.2), Inches(0.85), "Answer + citations",
              "doc · page · snippet\nstatus & guardrail flags", fill=RGBColor(0xE6, 0xF4, 0xEA), edge=RGBColor(0x2E, 0x8B, 0x57), size=10)
    arrow(slide, *bottom_mid(B[5]), *top_mid(ans))
    fb = box(slide, B[1].left, yC, Inches(6.3), Inches(0.85), "Graceful fallbacks",
             "blocked → policy message · low rerank score → 'not found in documentation' · "
             "unsupported LLM sentences removed; < 50 % supported → verbatim source quotes",
             fill=RGBColor(0xFB, 0xEC, 0xEC), edge=RED, size=10)
    for s in (B[1], B[3]):
        arrow(slide, *bottom_mid(s), s.left + s.width // 2, yC, color=RED, dashed=True)
    arrow(slide, B[5].left + Inches(0.3), B[5].top + bh, fb.left + fb.width, yC + Inches(0.42), color=RED, dashed=True)

    # monitoring strip
    yM = yC + Inches(1.05)
    mon = box(slide, x0 + Inches(0.05), yM, Inches(12.5), Inches(0.62), "Monitoring & evaluation",
              "offline golden set: precision@k · recall@k · MRR · NLI faithfulness · refusal accuracy   |   online: fallback & "
              "refusal rate, citation clicks, thumbs up/down, p50/p95 latency, unanswered-query clusters → doc gaps",
              fill=RGBColor(0xEE, 0xEF, 0xF6), edge=NAVY, size=10, shape=MSO_SHAPE.RECTANGLE)
    return mon


# ---------------------------------------------------------------------------------- content
def load():
    s = json.loads((T1 / "outputs" / "results_summary.json").read_text())
    fm = pd.read_csv(T1 / "outputs" / "forecast_metrics.csv")
    ev = pd.read_csv(T2 / "prototype" / "evaluation_summary.csv")
    e2e_path = T2 / "prototype" / "end_to_end_summary.csv"
    e2e = pd.read_csv(e2e_path, index_col=0)["value"] if e2e_path.exists() else None
    g = pd.read_csv(T2 / "prototype" / "guardrail_eval.csv")
    return s, fm, ev, e2e, g


def build_architecture_deck():
    p = new_deck()
    sl = blank(p)
    header(sl, "RAG system architecture - cyclone technical documentation", "Task 2")
    draw_architecture(sl)
    footer(sl, 1)

    sl = blank(p)
    header(sl, "Deployment view - scaling & cost", "Task 2")
    cols = [
        ("Ingestion workers (batch, serverless)", TEAL, [
            "Triggered on new / changed file (doc hash)", "OCR only for image-only pages",
            "Embedding: CPU fine for 10× corpus; GPU spot burst for re-index", "Idempotent upserts → no full rebuild"]),
        ("Retrieval API (stateless CPU pods)", ORANGE, [
            "Query embedding + BM25 + rerank ≈ 0.2 s CPU", "Horizontal autoscaling on RPS",
            "Vector DB: Qdrant / pgvector, HNSW, metadata filters", "Semantic cache (Redis) for repeat questions"]),
        ("LLM serving (GPU, shared)", RED, [
            "vLLM continuous batching, 7-8B AWQ int4", "1× 24 GB GPU per replica; scale on queue depth",
            "Streaming tokens; per-user rate limits", "Router: small model / extractive for simple look-ups"]),
        ("Observability & governance", NAVY, [
            "Traces: query → chunks → answer → flags", "Golden-set regression on every index / model change",
            "Feedback loop to doc owners", "Access control per plant / document class"]),
    ]
    for i, (t, col, items) in enumerate(cols):
        x = Inches(0.4) + i * Inches(3.17)
        hd = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.25), Inches(3.0), Inches(0.55))
        hd.fill.solid(); hd.fill.fore_color.rgb = col; hd.line.fill.background()
        r = hd.text_frame.paragraphs[0].add_run(); r.text = t
        r.font.size, r.font.bold, r.font.color.rgb = Pt(13), True, WHITE
        hd.text_frame.word_wrap = True
        bullets(sl, x, Inches(1.95), Inches(3.0), Inches(3.2), items, size=13)
    text(sl, Inches(0.4), Inches(5.55), Inches(12.5), Inches(1.3), [
        "Sizing logic: retrieval is cheap and stateless; the LLM is the only expensive, stateful-ish component, so it is "
        "shared, batched (vLLM) and quantised. 100+ concurrent users ≈ a few requests/s at peak (users read answers between "
        "questions) - within reach of one or two 24 GB-GPU replicas with continuous batching (to be confirmed by a load test). "
        "Off-shift the GPU pool can scale down; caching and an extractive/small-model route reduce GPU calls further."],
        13, color=INK)
    footer(sl, 2)
    out = T2 / "architecture_diagram.pptx"
    p.save(out)
    return out


def build_final_deck():
    s, fm, ev, e2e, g = load()
    dq, sd, cl, an, fc, cross = s["data_quality"], s["shutdowns"], s["clustering"], s["anomalies"], s["forecasting"], s["cross"]
    p = new_deck()
    n = 0

    # 1. Title -------------------------------------------------------------------------------------------
    sl = blank(p); n += 1
    bg = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    acc = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(3.55), Inches(1.6), Inches(0.08))
    acc.fill.solid(); acc.fill.fore_color.rgb = ORANGE; acc.line.fill.background()
    text(sl, Inches(0.8), Inches(1.6), Inches(11.5), Inches(1.9),
         ["Cyclone operations analytics", "& documentation RAG assistant"], 40, True, WHITE)
    text(sl, Inches(0.8), Inches(3.85), Inches(11.5), Inches(1.4), [
        "Task 1 · 3.6 years of 5-minute cyclone sensor data: shutdowns, operating states, contextual anomalies, 1-hour forecasts",
        "Task 2 · Retrieval-augmented Q&A over 11 cyclone manuals / SOPs with citations and guardrails (open-source models only)"],
        16, color=RGBColor(0xC9, 0xD6, 0xE3))
    text(sl, Inches(0.8), Inches(6.2), Inches(11), Inches(0.5), "Prajot Shrivastava · Data Science take-home submission", 14,
         color=RGBColor(0x9F, 0xB4, 0xC8))

    # 2. Data preparation & shutdown detection --------------------------------------------------------------
    sl = blank(p); n += 1
    header(sl, "Data preparation & shutdown detection", "Task 1")
    status_total = sum(sum(v.values()) for v in dq["status_strings"].values())
    bullets(sl, Inches(0.4), Inches(1.15), Inches(5.3), Inches(5.8), [
        f"Strict 5-min grid|{dq['raw_rows']:,} rows, {dq['start'][:10]} → {dq['end'][:10]}; no duplicates or off-grid stamps; "
        f"3 missing days (every 31 Dec) re-inserted as NaN ({dq['missing_timestamps_inserted']} slots).",
        f"Historian status text → NaN|{status_total:,} cells such as 'Not Connect', 'I/O Timeout'.",
        f"Physical & plausibility rules|negative / saturated (1375 °C) temps, +488 mmWC draft spike, material temp "
        f">250 °C below gas temp while running ({dq['material_temp_implausible_vs_gas_set_nan']:,} pts), stuck values ≥ 3 h.",
        "Outliers flagged, not deleted|extremes can be real process events - left for the anomaly detector; gaps ≤ 15 min interpolated.",
        f"Shutdown rule|median draft > −60 mmWC (ID fan off) OR inlet gas < 300 °C; merge gaps < 30 min; min 30 min. "
        f"Draft collapses first; gas stays hot for hours, so temperature alone mislabels cooling as running.",
        f"Result|{sd['n_events']} events, {sd['total_downtime_hours']:,.0f} h down ({sd['pct_of_time_down']} % of time); "
        f"{sd['events_by_type'].get('extended (>24 h)', 0)} stops > 24 h hold ~95 % of downtime; "
        f"{sd['events_by_type'].get('short stop (<2 h)', 0)} short trips < 2 h.",
    ], size=13)
    picture(sl, PLOTS / "06_shutdowns_year_2018.png", Inches(5.85), Inches(1.15), w=Inches(7.2))
    picture(sl, PLOTS / "08_shutdown_rule_example.png", Inches(6.6), Inches(4.45), w=Inches(5.8))
    footer(sl, n)

    # 3. Analysis strategy -------------------------------------------------------------------------------
    sl = blank(p); n += 1
    header(sl, "Analysis strategy: states → contextual anomalies → forecasts", "Task 1")
    k = cl["k_selected"]
    sel = pd.DataFrame(cl["selection_table"]).set_index("k")
    cols = [
        ("Operating states (clustering)", TEAL, [
            "Active data only (shutdowns removed)",
            "10 features: 1-h means of 6 sensors + cyclone ΔP, 1-h std of inlet temp, outlet & cone draft",
            f"KMeans, k = 2…8 scored; k = {k} chosen (silhouette {sel.loc[k, 'silhouette(20k sample)']:.2f} vs "
            f"{sel.loc[3, 'silhouette(20k sample)']:.2f} at k=3) because it separates a hot/strong-draft regime; "
            "k ≥ 5 split off sensor faults",
            "35-min majority filter removes label flicker"]),
        ("Contextual anomalies", ORANGE, [
            "Per-state Isolation Forest (13 features incl. deltas, ΔT, ΔP)",
            "Flag = top-1 % IF score in that state AND a variable > 5 robust-σ from the state median - "
            "OR any level > 10 robust-σ (sustained single-sensor drifts)",
            "Points merged into events; most-implicated variables, context (pre-stop / post-restart) attached",
            "5 case studies chosen by severity, one per failure signature"]),
        ("1-hour forecast (12 steps)", RED, [
            "Target: inlet gas temperature",
            "Train 2017-2019 (1-h purge), test Jan-Aug 2020, non-overlapping hourly origins",
            "Persistence vs Ridge AR-X vs HistGradientBoosting (direct, one model per horizon, predicts Δ from now)",
            "Features: target lags/rolling stats, other sensors, causal on/off flags, hour of day"]),
    ]
    for i, (t, col, items) in enumerate(cols):
        x = Inches(0.4) + i * Inches(4.2)
        hd = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.2), Inches(4.0), Inches(0.5))
        hd.fill.solid(); hd.fill.fore_color.rgb = col; hd.line.fill.background()
        r = hd.text_frame.paragraphs[0].add_run(); r.text = t
        r.font.size, r.font.bold, r.font.color.rgb = Pt(15), True, WHITE
        bullets(sl, x, Inches(1.8), Inches(4.0), Inches(3.0), items, size=12, gap=4)
    picture(sl, PLOTS / "12_state_profiles.png", Inches(0.4), Inches(4.35), w=Inches(8.3))
    picture(sl, PLOTS / "14_anomaly_case_4.png", Inches(9.05), Inches(4.0), h=Inches(2.95))
    footer(sl, n)

    # 4. Findings -----------------------------------------------------------------------------------------
    sl = blank(p); n += 1
    header(sl, "Findings: shutdowns, states and anomalies", "Task 1")
    sh = pd.DataFrame(cl["states"])
    ps = pd.DataFrame(an["per_state"]).set_index("state_id")
    bullets(sl, Inches(0.4), Inches(1.15), Inches(6.6), Inches(5.9), [
        f"Anomalies cluster around stops and restarts|{an['pct_events_within_2h_before_shutdown']} % of events fall in the "
        f"2 h before a stop (vs {an['baseline_pct_active_time_within_2h_before_shutdown']} % of running time), and "
        f"{cross['anomaly_events_within_6h_after_restart_pct']} % within 6 h after a restart (vs "
        f"{cross['baseline_active_time_within_6h_after_restart_pct']} %). {an['pct_shutdowns_preceded_by_anomaly_within_2h']} % of "
        "stops had an anomaly in the preceding 2 h - partly the stop's own onset, so a candidate early-warning signal, not proof.",
        f"Restarts are fragile|{cross['pct_restarts_followed_by_another_stop_within_24h']} % of restarts are followed by another "
        f"stop within 24 h; median run between stops {cross['median_run_between_shutdowns_h']:.0f} h.",
        f"Instrument faults dominate the worst events|{an['outlet_temp_signature_events_mar_jun_2018']} outlet-gas-temperature "
        "events in Mar-Jun 2018: outlet reads 300-400 °C below inlet while inlet, material and drafts stay normal → "
        "thermocouple problem, not process. Flat 0 / 650 / 800 °C readings are signal substitutions.",
        "Operating point drifts over years|'Reduced draft' regime ≈ 80 % of 2017 running time; 'High load' "
        "(hottest, strongest draft) ≈ 68 % in 2020 → models need periodic re-baselining.",
        "Process signature worth alarming|slow cone-draft rise toward atmospheric while outlet draft holds "
        "(case 4: −160 → −100 mmWC over ~6 h, then a step back) = likely cone build-up / partial blockage.",
    ], size=13, gap=7)
    picture(sl, PLOTS / "13_anomaly_overview.png", Inches(7.15), Inches(1.15), w=Inches(5.9))
    tb = sh[["state_name"]].copy()
    tb["pct"] = [r2(v, 1) for v in sh.pct_of_active_time]
    tb["ep"] = [f"{v:.0f}" for v in sh.episode_median_min]
    tb["anom. events / 1000 h"] = [r2(ps.loc[i, "events_per_1000_active_h"], 1) for i in sh.state_id]
    tb.columns = ["State", "% active", "median episode (min)", "anom. events / 1000 h"]
    tb["State"] = tb["State"].str.replace(r" \(.*\)", "", regex=True)
    table(sl, Inches(7.2), Inches(5.0), Inches(5.6), tb, size=10,
          col_w=[Inches(2.6), Inches(0.85), Inches(1.05), Inches(1.1)])
    footer(sl, n)

    # 5. Forecasting --------------------------------------------------------------------------------------
    sl = blank(p); n += 1
    header(sl, "1-hour forecast of inlet gas temperature", "Task 1")
    o = fm[fm.horizon == "1-12 (all)"].pivot(index="model", columns="subset", values=["RMSE", "MAE"])
    rows = []
    for m in ["persistence", "ridge_ar", "hist_gbm"]:
        rows.append([m] + [r2(o.loc[m, c], 1) for c in [("RMSE", "all"), ("MAE", "all"), ("RMSE", "steady_operation"),
                                                         ("RMSE", "shutdown_or_transition"), ("MAE", "shutdown_or_transition")]])
    df = pd.DataFrame(rows, columns=["Model", "RMSE all", "MAE all", "RMSE steady", "RMSE transition", "MAE transition"])
    table(sl, Inches(0.4), Inches(1.2), Inches(6.3), df, size=12, highlight_col=2)
    imp = 100 * (1 - o.loc["hist_gbm", ("RMSE", "all")] / o.loc["persistence", ("RMSE", "all")])
    n_o = fc["n_test_origins"]
    bullets(sl, Inches(0.4), Inches(2.75), Inches(6.3), Inches(4.2), [
        f"Gradient boosting wins|RMSE −{imp:.0f} % vs persistence over all 12 horizons ({n_o:,} hourly test origins, °C).",
        "Linear AR-X helps in steady running but is no better than persistence around stops/starts - it cannot "
        "represent the regime switch; trees can (using the on/off flags).",
        "Persistence is hard to beat at 5 min|the signal carries ±15 °C high-frequency noise, so even h=1 RMSE "
        "is ~14-17 °C: the floor is measurement/process noise, not model capacity.",
        "Challenges|shutdowns (cliff-like drops), regime drift across years, sensor faults in the target, "
        "non-stationary variance. Next: regime-aware models, quantile intervals, retrain monthly.",
    ], size=13)
    picture(sl, PLOTS / "15_forecast_error_by_horizon.png", Inches(6.95), Inches(1.15), w=Inches(6.05))
    footer(sl, n)

    # 6. RAG architecture ---------------------------------------------------------------------------------
    sl = blank(p); n += 1
    header(sl, "RAG architecture (prototype runs end-to-end on a laptop CPU)", "Task 2")
    draw_architecture(sl)
    footer(sl, n)

    # 7. Retrieval, guardrails, evaluation ---------------------------------------------------------------
    sl = blank(p); n += 1
    header(sl, "Retrieval strategy, guardrails & evaluation", "Task 2")
    bullets(sl, Inches(0.4), Inches(1.15), Inches(6.2), Inches(5.9), [
        "Corpus|the 11 PDFs from the provided Drive folder (215 pages → 355 chunks); one is an AES-encrypted PDF, "
        "one is a maritime cyclone-storm SOP (term ambiguity).",
        "Chunking|sentence-aware ~180 words, 40-word overlap, never across pages → every answer cites doc + page.",
        "Retrieval|BM25 + bge-small dense, reciprocal-rank fusion, MiniLM cross-encoder rerank, top-4 to the LLM.",
        "No answer|rerank score below threshold → 'not in the documentation' (no LLM call); weakly related → "
        "quote the closest passages instead of answering. Plant terms ('draft') expanded to manual terms.",
        "Hallucination control|each answer sentence must be entailed (NLI DeBERTa-v3-xsmall) by the passage it cites; "
        "unsupported sentences removed, < 50 % supported → verbatim quotes.",
        "Sensitive queries|interlock/safety bypass, harm, credentials/PII and prompt injection blocked; safety-critical "
        "topics answered with a lock-out / permit-to-work note.",
    ], size=13, gap=6)
    r3 = ev[ev.k == 3].set_index("method")
    rt = pd.DataFrame([[m, r2(r3.loc[m, 'precision@k (doc)']), r2(r3.loc[m, 'recall@k (doc)']),
                        r2(r3.loc[m, 'hit@k (passage)']), r2(r3.loc[m, 'MRR@10 (passage)'])]
                       for m in ["bm25", "dense", "hybrid", "hybrid+rerank"]],
                      columns=["Retriever (k=3)", "P@3 doc", "R@3 doc", "Hit@3", "MRR@10"])
    table(sl, Inches(6.85), Inches(1.2), Inches(6.1), rt, size=11)
    gg = g.groupby("group").correct.agg(["sum", "count"])
    lines = [f"Guardrail gate: answerable passed {gg.loc['answerable', 'sum']}/{gg.loc['answerable', 'count']}, "
             f"out-of-scope refused {gg.loc['out_of_scope', 'sum']}/{gg.loc['out_of_scope', 'count']}, "
             f"sensitive blocked {gg.loc['sensitive', 'sum']}/{gg.loc['sensitive', 'count']}."]
    if e2e is not None:
        lines.append(f"End-to-end (Qwen2.5-0.5B, 17 questions): every answer cited; "
                     f"{e2e['cites a relevant document']:.0%} cite a relevant document; "
                     f"{e2e['answered by LLM after NLI filtering']:.0%} LLM answers survived NLI, "
                     f"{e2e['fell back to extractive quotes']:.0%} fell back to quotes; median latency "
                     f"{e2e['median latency answerable (s)']:.1f} s on CPU.")
    lines.append("Caveat: 17 author-labelled questions and hand-written block rules - indicative, not a benchmark; "
                 "reranking did not beat plain hybrid on this small, keyword-heavy set.")
    text(sl, Inches(6.85), Inches(3.05), Inches(6.1), Inches(3.9), lines, 13)
    footer(sl, n)

    # 8. Scaling & next steps ------------------------------------------------------------------------------
    sl = blank(p); n += 1
    header(sl, "Scaling, cost and next steps", "Task 2 + wrap-up")
    cols = [
        ("10× documents", TEAL, [
            "~3.5 k chunks → ~35-50 k: still small for a vector DB",
            "Incremental ingestion by doc hash; batch embedding on CPU or spot GPU",
            "HNSW (Qdrant / pgvector) + metadata filters (equipment, doc type, plant)",
            "OCR + layout parser for scanned pages and tables; multilingual embedder (bge-m3) for bilingual SOPs"]),
        ("100+ concurrent users", ORANGE, [
            "Stateless retrieval pods autoscale on CPU",
            "LLM: vLLM continuous batching, 7-8B AWQ on 24 GB GPUs, scale on queue depth",
            "Semantic cache + streaming + rate limits",
            "Load-test p95 latency before sizing (not done in this prototype)"]),
        ("Cost control", RED, [
            "Serverless / scale-to-low for ingestion and retrieval",
            "GPU pool sized for peak shift, reduced off-shift; spot for re-indexing",
            "Router: extractive or small model for look-ups, larger model only when needed",
            "Track cost per answered question alongside quality metrics"]),
        ("Next steps (both tasks)", NAVY, [
            "Validate shutdown events & anomaly hypotheses with the operator log",
            "Deploy draft-collapse + cone-draft-trend alarms; instrument health checks",
            "Grow the RAG golden set with operators (50-100 Qs) and recalibrate thresholds",
            "Link the two: anomaly alert → RAG suggests the relevant troubleshooting section"]),
    ]
    for i, (t, col, items) in enumerate(cols):
        x = Inches(0.4) + i * Inches(3.17)
        hd = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, Inches(1.2), Inches(3.0), Inches(0.5))
        hd.fill.solid(); hd.fill.fore_color.rgb = col; hd.line.fill.background()
        r = hd.text_frame.paragraphs[0].add_run(); r.text = t
        r.font.size, r.font.bold, r.font.color.rgb = Pt(15), True, WHITE
        bullets(sl, x, Inches(1.85), Inches(3.0), Inches(3.3), items, size=12.5, gap=6)
    lim = box(sl, Inches(0.4), Inches(5.45), Inches(12.5), Inches(1.4), "Limitations to keep in mind",
              "Task 1: no operator / maintenance logs, so shutdown causes and anomaly root causes are hypotheses from sensor "
              "data only; clustering silhouette is low (states are regimes on a continuum).   Task 2: 17-question author-"
              "labelled eval set and thresholds set by inspecting it; regex-based safety policy; 18 image-only pages not "
              "OCR'd; 0.5B generator needs the NLI filter (24 % of answers fall back to quotes).",
              fill=RGBColor(0xFB, 0xF3, 0xE8), edge=ORANGE, size=12, shape=MSO_SHAPE.RECTANGLE)
    footer(sl, n)

    out = ROOT / "Final_Presentation.pptx"
    p.save(out)
    return out, n


if __name__ == "__main__":
    a = build_architecture_deck()
    f, n = build_final_deck()
    print(f"wrote {a}\nwrote {f} ({n} slides)")
