#!/usr/bin/env python3
"""
Task 1 - Cyclone machine time-series analysis (end-to-end, reproducible).

Pipeline
  1. Load + data-quality checks, strict 5-minute index, cleaning, EDA plots
  2. Shutdown / idle detection (draft-collapse OR low inlet temperature rule)
  3. Operational-state segmentation (KMeans on engineered features, active data only)
  4. State-contextual anomaly detection (per-state Isolation Forest + robust z consensus)
     + automatic evidence extraction for selected case studies
  5. 1-hour (12-step) forecasting of Cyclone_Inlet_Gas_Temp: persistence vs Ridge vs
     HistGradientBoosting, time-ordered hold-out test
  6. Key numbers written to outputs/results_summary.json (used for README insights)

Run:  python task1_analysis.py                (defaults: data/data.xlsx -> ./ outputs)
      python task1_analysis.py --data /path/to/data.xlsx
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------
# Configuration (every tunable decision lives here and is documented in README.md)
# --------------------------------------------------------------------------------------
SEED = 42
FREQ = "5min"
SHEET = "internship-data-1"
VARS = [
    "Cyclone_Inlet_Gas_Temp",
    "Cyclone_Gas_Outlet_Temp",
    "Cyclone_Material_Temp",
    "Cyclone_Inlet_Draft",
    "Cyclone_cone_draft",
    "Cyclone_Outlet_Gas_draft",
]
SHORT = {
    "Cyclone_Inlet_Gas_Temp": "InletGasT",
    "Cyclone_Gas_Outlet_Temp": "OutletGasT",
    "Cyclone_Material_Temp": "MaterialT",
    "Cyclone_Inlet_Draft": "InletDraft",
    "Cyclone_cone_draft": "ConeDraft",
    "Cyclone_Outlet_Gas_draft": "OutletDraft",
}
TEMPS = VARS[:3]
DRAFTS = VARS[3:]

# Cleaning
TEMP_VALID = (0.0, 1370.0)        # <0 is physically impossible; 1375 is an instrument range ceiling
DRAFT_VALID = (-600.0, 100.0)     # drafts are negative pressures; +488 cone draft is a spike artefact
FLATLINE_MIN_POINTS = 36          # identical value for >= 3 h => stuck / held sensor value
MAT_FAULT_GAP = 250.0             # material temp > 250 C below inlet gas temp while running => sensor fault
INTERP_LIMIT = 3                  # linearly interpolate gaps of <= 15 min only

# Shutdown detection
DRAFT_OFF = -60.0                 # median draft above this => ID fan off / no gas flow
TEMP_OFF = 300.0                  # inlet gas temperature below this => no hot gas
MIN_SHUTDOWN_POINTS = 6           # ignore "down" runs shorter than 30 min (dips, not shutdowns)
MERGE_GAP_POINTS = 6              # merge shutdowns separated by < 30 min of "running"
UNKNOWN_FILL_POINTS = 12          # carry state across sensor-data gaps of <= 1 h

# Clustering
K_RANGE = range(2, 9)
K_FINAL = 4                       # fixed after reviewing cluster_model_selection.csv + profiles (README §3);
                                  # set to None to use the automatic silhouette rule in choose_k()
MODE_FILTER_POINTS = 7            # 35-min centred majority filter to remove label flicker

# Anomaly detection
IF_TOP_QUANTILE = 0.99            # per-state Isolation-Forest score must be in the state's top 1 %
ROBUST_Z_MIN = 5.0                # ...AND at least one variable > 5 robust-sigma from the state median
ROBUST_Z_STRONG = 10.0            # OR any single level variable > 10 robust-sigma (sustained single-sensor shifts
                                  #    that a multivariate IF under-scores, e.g. a drifting thermocouple)
EVENT_MERGE_POINTS = 3            # merge flagged points separated by <= 15 min into one event
N_CASE_STUDIES = 5

# Forecasting
TARGET = "Cyclone_Inlet_Gas_Temp"
HORIZON = 12
TEST_START = "2020-01-01"         # time-ordered hold-out: train 2017-2019, test 2020 (~7 months)
TRAIN_STRIDE = 2                  # use every 2nd origin in training (speed; adjacent rows ~duplicate)

STATE_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def runs(mask: pd.Series) -> pd.DataFrame:
    """Contiguous True-runs of a boolean series on a regular index -> start/end positions."""
    m = mask.fillna(False).to_numpy().astype(int)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]  # end is exclusive
    return pd.DataFrame({"i0": starts, "i1": ends, "n": ends - starts})


def savefig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 1. Load, validate, clean, EDA
# --------------------------------------------------------------------------------------
def load_and_clean(data_path: Path, out: Path, plots: Path, summary: dict):
    log(f"Loading {data_path} ...")
    raw = pd.read_excel(data_path, sheet_name=SHEET)
    q = {"raw_rows": int(len(raw)), "raw_columns": list(raw.columns)}

    raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    q["unparseable_timestamps"] = int(raw["time"].isna().sum())
    raw = raw.dropna(subset=["time"])
    q["duplicate_timestamps"] = int(raw["time"].duplicated().sum())
    raw = raw.drop_duplicates("time", keep="first").sort_values("time").set_index("time")
    off_grid = raw.index[(raw.index - raw.index.floor(FREQ)) != pd.Timedelta(0)]
    q["off_grid_timestamps"] = int(len(off_grid))
    raw.index = raw.index.floor(FREQ)

    # Non-numeric historian status strings ("Not Connect", "I/O Timeout", ...)
    status_counts = {}
    num = pd.DataFrame(index=raw.index)
    for c in VARS:
        s = raw[c]
        conv = pd.to_numeric(s, errors="coerce")
        bad = s[conv.isna() & s.notna()].astype(str)
        status_counts[c] = bad.value_counts().to_dict()
        num[c] = conv.astype(float)
    q["status_strings"] = status_counts

    # Strict 5-minute grid
    full_idx = pd.date_range(num.index.min(), num.index.max(), freq=FREQ, name="time")
    q["expected_rows_on_5min_grid"] = int(len(full_idx))
    df = num.reindex(full_idx)
    missing_ts = df.index.difference(num.index)
    q["missing_timestamps_inserted"] = int(len(missing_ts))
    gap_runs = runs(pd.Series(df.index.isin(missing_ts), index=df.index))
    q["timestamp_gaps"] = [
        {"start": str(df.index[r.i0]), "end": str(df.index[r.i1 - 1]), "points": int(r.n)}
        for r in gap_runs.itertuples()
    ]
    q["start"], q["end"] = str(df.index.min()), str(df.index.max())
    q["span_days"] = round((df.index.max() - df.index.min()) / pd.Timedelta("1D"), 1)

    raw_numeric = df.copy()  # numeric, gridded, before outlier handling (for EDA)

    # Physical-range validity + saturation
    invalid = {}
    for c in VARS:
        lo, hi = TEMP_VALID if c in TEMPS else DRAFT_VALID
        bad = (df[c] < lo) | (df[c] > hi)
        invalid[c] = int(bad.sum())
        df.loc[bad, c] = np.nan
    q["out_of_physical_range_set_nan"] = invalid

    # Cross-sensor plausibility: material temp << gas temp while gas is hot and fan is running
    # (material leaves the cyclone near gas temperature) => material-temperature sensor fault
    mat_fault = ((df["Cyclone_Material_Temp"] < df["Cyclone_Inlet_Gas_Temp"] - MAT_FAULT_GAP)
                 & (df["Cyclone_Inlet_Gas_Temp"] > 750)
                 & (df[DRAFTS].median(axis=1) < DRAFT_OFF))
    q["material_temp_implausible_vs_gas_set_nan"] = int(mat_fault.sum())
    df.loc[mat_fault, "Cyclone_Material_Temp"] = np.nan

    # Flat-lined (stuck / held) sensor values
    flat = {}
    for c in VARS:
        s = df[c]
        same = s.eq(s.shift()) & s.notna()
        grp = (~same).cumsum()
        size = s.groupby(grp).transform("size")
        stuck = (size >= FLATLINE_MIN_POINTS) & s.notna()
        flat[c] = int(stuck.sum())
        df.loc[stuck, c] = np.nan
    q["flatline_points_set_nan"] = flat

    # Statistical outliers are *flagged* (not removed): extremes may be genuine process events
    stat_out = {}
    for c in VARS:
        s = df[c]
        med = s.rolling(13, center=True, min_periods=5).median()
        mad = (s - med).abs().rolling(289, center=True, min_periods=50).median()
        z = (s - med).abs() / (1.4826 * mad.replace(0, np.nan))
        stat_out[c] = int((z > 10).sum())
    q["statistical_spikes_flagged_not_removed(hampel_z>10)"] = stat_out

    q["missing_after_cleaning_before_interp"] = {c: int(df[c].isna().sum()) for c in VARS}
    clean = df.interpolate(method="time", limit=INTERP_LIMIT, limit_area="inside")
    q["missing_after_interp(<=15min)"] = {c: int(clean[c].isna().sum()) for c in VARS}
    q["rows_all_vars_present_after_interp"] = int(clean.notna().all(axis=1).sum())

    summary["data_quality"] = q
    with open(out / "data_quality_report.json", "w") as f:
        json.dump(q, f, indent=2, default=str)

    # Missingness per month plot
    fig, ax = plt.subplots(figsize=(12, 3.5))
    miss = df.isna().resample("MS").mean() * 100
    miss.columns = [SHORT[c] for c in miss.columns]
    miss.plot(ax=ax, marker="o", ms=3)
    ax.set_ylabel("% missing (after cleaning)")
    ax.set_title("Missing / invalid data by month (status strings, out-of-range, stuck values, gaps)")
    ax.legend(ncol=6, fontsize=8)
    savefig(fig, plots / "01_missing_by_month.png")

    # Distributions (raw numeric)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, c in zip(axes.ravel(), VARS):
        raw_numeric[c].dropna().plot.hist(bins=120, ax=ax, color="#4c72b0", log=True)
        ax.set_title(c, fontsize=9)
    fig.suptitle("Raw value distributions (log count) - bimodality = running vs shutdown")
    savefig(fig, plots / "02_distributions.png")
    return clean


def eda(clean: pd.DataFrame, down: pd.Series, out: Path, plots: Path, summary: dict):
    active = clean[~down]
    stats_all = clean.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats_act = active.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    stats_all.insert(0, "subset", "all_data")
    stats_act.insert(0, "subset", "active_only")
    pd.concat([stats_all, stats_act]).round(3).rename_axis("variable").to_csv(out / "summary_statistics.csv")

    corr_all = clean.corr()
    corr_act = active.corr()
    corr_all.round(3).to_csv(out / "correlation_matrix_all.csv")
    corr_act.round(3).to_csv(out / "correlation_matrix_active.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cm, title in [(axes[0], corr_all, "All data (dominated by on/off)"),
                          (axes[1], corr_act, "Active operation only")]:
        im = ax.imshow(cm.values, cmap="RdBu_r", vmin=-1, vmax=1)
        lbl = [SHORT[c] for c in cm.columns]
        ax.set_xticks(range(6), lbl, rotation=45, ha="right")
        ax.set_yticks(range(6), lbl)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"Pearson correlation - {title}")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(plots / "03_correlation_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    summary["eda"] = {
        "corr_active_inlet_vs_outlet_gas_temp": round(float(corr_act.loc[VARS[0], VARS[1]]), 3),
        "corr_active_inlet_temp_vs_material_temp": round(float(corr_act.loc[VARS[0], VARS[2]]), 3),
        "corr_active_inlet_draft_vs_outlet_draft": round(float(corr_act.loc[VARS[3], VARS[5]]), 3),
        "corr_active_cone_vs_outlet_draft": round(float(corr_act.loc[VARS[4], VARS[5]]), 3),
        "corr_all_inlet_temp_vs_outlet_draft": round(float(corr_all.loc[VARS[0], VARS[5]]), 3),
        "active_medians": active.median().round(1).to_dict(),
    }

    # Representative week: pick the first full week of 2018 with no shutdown and little missing data
    best = None
    for start in pd.date_range("2018-07-02", "2019-12-01", freq="7D"):
        sl = slice(start, start + pd.Timedelta("7D") - pd.Timedelta(FREQ))
        if down[sl].sum() == 0 and clean.loc[sl].isna().mean().max() < 0.01:
            best = start
            break
    best = best if best is not None else pd.Timestamp("2018-07-02")
    wk = clean.loc[best: best + pd.Timedelta("7D")]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for c in TEMPS:
        axes[0].plot(wk.index, wk[c], lw=0.8, label=SHORT[c])
    for c in DRAFTS:
        axes[1].plot(wk.index, wk[c], lw=0.8, label=SHORT[c])
    axes[0].set_ylabel("Temperature (°C)")
    axes[1].set_ylabel("Draft (mmWC, negative = suction)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].legend(ncol=3, fontsize=8)
    axes[0].set_title(f"Representative week of normal operation ({best.date()} +7 d, no shutdowns)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d-%b"))
    savefig(fig, plots / "04_week_slice.png")
    summary["eda"]["week_slice_start"] = str(best.date())

    # Representative year: daily median with 5-95 % band, 2018
    yr = clean.loc["2018"]
    fig, axes = plt.subplots(len(VARS), 1, figsize=(14, 13), sharex=True)
    for ax, c in zip(axes, VARS):
        d = yr[c].resample("D")
        ax.fill_between(d.quantile(0.05).index, d.quantile(0.05), d.quantile(0.95), alpha=0.3,
                        label="daily 5-95 %")
        ax.plot(d.median().index, d.median(), lw=1, label="daily median")
        ax.set_ylabel(SHORT[c], fontsize=8)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title("Representative year (2018): daily median and 5-95 % band - dips are shutdowns")
    savefig(fig, plots / "05_year_slice_2018.png")


# --------------------------------------------------------------------------------------
# 2. Shutdown / idle detection
# --------------------------------------------------------------------------------------
def detect_shutdowns(clean: pd.DataFrame, out: Path, plots: Path, summary: dict):
    med_draft = clean[DRAFTS].median(axis=1, skipna=True)        # robust to one faulty draft sensor
    draft_off = med_draft > DRAFT_OFF
    temp_off = clean[TARGET] < TEMP_OFF
    has_draft = med_draft.notna()
    has_temp = clean[TARGET].notna()
    known = has_draft | has_temp
    point_down = (draft_off & has_draft) | (temp_off & has_temp)

    # Unknown points (no data): carry previous state across short gaps only
    state = pd.Series(np.where(known, point_down.astype(float), np.nan), index=clean.index)
    state = state.ffill(limit=UNKNOWN_FILL_POINTS)
    unknown = state.isna()
    down = state.fillna(0).astype(bool)

    # 1) Close short "running" gaps inside shutdowns (merge), 2) drop short down runs
    up_runs = runs(~down & ~unknown)
    arr = down.to_numpy().copy()
    for r in up_runs.itertuples():
        if r.n < MERGE_GAP_POINTS and r.i0 > 0 and r.i1 < len(arr) and arr[r.i0 - 1] and arr[r.i1]:
            arr[r.i0: r.i1] = True
    down = pd.Series(arr, index=clean.index)
    for r in runs(down).itertuples():
        if r.n < MIN_SHUTDOWN_POINTS:
            arr[r.i0: r.i1] = False
    down = pd.Series(arr, index=clean.index)

    ev = runs(down)
    rows = []
    for k, r in enumerate(ev.itertuples(), 1):
        seg = clean.iloc[r.i0: r.i1]
        start = clean.index[r.i0]
        end = clean.index[r.i1 - 1] + pd.Timedelta(FREQ)  # exclusive end
        trig_d = (draft_off.iloc[r.i0: r.i1] & has_draft.iloc[r.i0: r.i1]).mean()
        trig_t = (temp_off.iloc[r.i0: r.i1] & has_temp.iloc[r.i0: r.i1]).mean()
        rows.append({
            "event_id": k,
            "start": start, "end": end,
            "duration_minutes": int(r.n * 5),
            "duration_hours": round(r.n * 5 / 60, 2),
            "min_inlet_gas_temp": round(float(seg[TARGET].min()), 1) if seg[TARGET].notna().any() else np.nan,
            "pct_points_draft_off": round(100 * trig_d, 1),
            "pct_points_temp_below_300C": round(100 * trig_t, 1),
            "contains_data_gap": bool(unknown.iloc[r.i0: r.i1].any() or seg.isna().all(axis=1).any()),
            "type": "short stop (<2 h)" if r.n * 5 < 120 else ("extended (>24 h)" if r.n * 5 > 1440 else "stop (2-24 h)"),
        })
    sd = pd.DataFrame(rows)
    sd.to_csv(out / "shutdown_periods.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    total_min = int(sd["duration_minutes"].sum()) if len(sd) else 0
    span_min = len(clean) * 5
    s = {
        "rule": f"median(drafts) > {DRAFT_OFF} mmWC OR inlet gas temp < {TEMP_OFF} C; "
                f"merge gaps < {MERGE_GAP_POINTS * 5} min; min duration {MIN_SHUTDOWN_POINTS * 5} min",
        "n_events": int(len(sd)),
        "total_downtime_hours": round(total_min / 60, 1),
        "total_downtime_days": round(total_min / 1440, 1),
        "pct_of_time_down": round(100 * total_min / span_min, 2),
        "unknown_state_hours(data gaps)": round(unknown.sum() * 5 / 60, 1),
        "median_duration_hours": round(float(sd["duration_hours"].median()), 2) if len(sd) else 0,
        "mean_duration_hours": round(float(sd["duration_hours"].mean()), 2) if len(sd) else 0,
        "max_duration_hours": round(float(sd["duration_hours"].max()), 2) if len(sd) else 0,
        "events_by_type": sd["type"].value_counts().to_dict() if len(sd) else {},
        "events_per_year": sd.groupby(sd["start"].dt.year).size().to_dict() if len(sd) else {},
        "downtime_hours_per_year": (sd.groupby(sd["start"].dt.year)["duration_minutes"].sum() / 60).round(1).to_dict() if len(sd) else {},
    }
    summary["shutdowns"] = s
    log(f"Shutdowns: {s['n_events']} events, {s['total_downtime_hours']} h ({s['pct_of_time_down']} %)")

    # Year plot with shutdowns highlighted (one per full year; 2018 is the headline plot)
    for year in ["2017", "2018", "2019"]:
        yr = clean.loc[year]
        fig, axes = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
        axes[0].plot(yr.index, yr[TARGET], lw=0.4, color="#333")
        axes[0].set_ylabel("Inlet gas temp (°C)")
        axes[1].plot(yr.index, yr["Cyclone_Outlet_Gas_draft"], lw=0.4, color="#1f77b4")
        axes[1].set_ylabel("Outlet gas draft (mmWC)")
        yev = sd[(sd["start"] < f"{int(year) + 1}-01-01") & (sd["end"] > f"{year}-01-01")]
        for ax in axes:
            for e in yev.itertuples():
                ax.axvspan(max(e.start, yr.index[0]), min(e.end, yr.index[-1]), color="red", alpha=0.25, lw=0)
        hrs = yev["duration_minutes"].sum() / 60
        axes[0].set_title(f"{year}: {len(yev)} shutdown/idle events highlighted in red "
                          f"({hrs:.0f} h downtime)")
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        savefig(fig, plots / f"06_shutdowns_year_{year}.png")

    # Monthly downtime + duration distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    dmin = down.resample("MS").sum() * 5 / 60
    axes[0].bar(dmin.index, dmin.values, width=20, color="#c44e52")
    axes[0].set_ylabel("Downtime (h)")
    axes[0].set_title("Monthly downtime")
    axes[1].hist(sd["duration_hours"], bins=np.logspace(np.log10(0.5), np.log10(max(sd["duration_hours"].max(), 1)), 30),
                 color="#c44e52")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Event duration (h, log scale)")
    axes[1].set_title("Shutdown duration distribution")
    savefig(fig, plots / "07_downtime_monthly_and_durations.png")

    # Zoom on one example event to justify the rule
    if len(sd):
        e = sd.iloc[len(sd) // 2]
        w = clean.loc[e.start - pd.Timedelta("6h"): e.end + pd.Timedelta("6h")]
        if len(w) > 2000:
            w = clean.loc[e.start - pd.Timedelta("6h"): e.start + pd.Timedelta("18h")]
        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(w.index, w[TARGET], color="#d62728", label="Inlet gas temp")
        ax1.axhline(TEMP_OFF, color="#d62728", ls=":", lw=1)
        ax2 = ax1.twinx()
        ax2.plot(w.index, w[DRAFTS].median(axis=1), color="#1f77b4", label="median draft")
        ax2.axhline(DRAFT_OFF, color="#1f77b4", ls=":", lw=1)
        ax1.axvspan(e.start, min(e.end, w.index[-1]), color="grey", alpha=0.2)
        ax1.set_ylabel("°C")
        ax2.set_ylabel("mmWC")
        ax1.set_title(f"Example shutdown #{e.event_id}: draft collapses first, temperature decays slowly "
                      "(dotted = thresholds)")
        fig.legend(loc="upper right", fontsize=8)
        savefig(fig, plots / "08_shutdown_rule_example.png")
    return down, unknown, sd


# --------------------------------------------------------------------------------------
# 3. Operational-state clustering
# --------------------------------------------------------------------------------------
def build_state_features(clean: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=clean.index)
    for c in VARS:
        f[SHORT[c]] = clean[c]
    # Physically motivated derived features
    f["GasTempDrop"] = clean["Cyclone_Inlet_Gas_Temp"] - clean["Cyclone_Gas_Outlet_Temp"]
    f["DraftLoss"] = clean["Cyclone_Inlet_Draft"] - clean["Cyclone_Outlet_Gas_draft"]  # pressure drop across cyclone
    f["ConeMinusInlet"] = clean["Cyclone_cone_draft"] - clean["Cyclone_Inlet_Draft"]
    # Short-term dynamics (trailing windows => usable online)
    f["InletGasT_d15"] = clean["Cyclone_Inlet_Gas_Temp"].diff(3)
    f["InletGasT_std1h"] = clean["Cyclone_Inlet_Gas_Temp"].rolling(12, min_periods=8).std()
    f["OutletDraft_std1h"] = clean["Cyclone_Outlet_Gas_draft"].rolling(12, min_periods=8).std()
    f["ConeDraft_std1h"] = clean["Cyclone_cone_draft"].rolling(12, min_periods=8).std()
    # 1-h trailing means: operating *regimes* rather than 5-min noise (used for clustering)
    for c in ["InletGasT", "OutletGasT", "MaterialT", "InletDraft", "ConeDraft", "OutletDraft", "DraftLoss"]:
        f[f"{c}_m1h"] = f[c].rolling(12, min_periods=8).mean()
    return f


CLUSTER_FEATURES = ["InletGasT_m1h", "OutletGasT_m1h", "MaterialT_m1h", "InletDraft_m1h", "ConeDraft_m1h",
                    "OutletDraft_m1h", "DraftLoss_m1h", "InletGasT_std1h", "OutletDraft_std1h", "ConeDraft_std1h"]
PROFILE_FEATURES = ["InletGasT", "OutletGasT", "MaterialT", "InletDraft", "ConeDraft", "OutletDraft",
                    "DraftLoss", "InletGasT_std1h", "OutletDraft_std1h", "ConeDraft_std1h"]


def mode_filter(labels: pd.Series, k: int, width: int) -> pd.Series:
    lab = labels.to_numpy()
    onehot = np.zeros((len(lab), k))
    ok = lab >= 0
    onehot[np.where(ok)[0], lab[ok].astype(int)] = 1
    sm = pd.DataFrame(onehot, index=labels.index).rolling(width, center=True, min_periods=1).sum().to_numpy()
    res = np.where(ok, sm.argmax(axis=1), -1)
    return pd.Series(res, index=labels.index)


def name_states(prof: pd.DataFrame) -> tuple[dict, dict]:
    """Deterministic rule-based naming from state medians (works for any k)."""
    names, desc = {}, {}
    gm = prof.median()
    unstable = [s for s in prof.index if prof.loc[s, "InletGasT_std1h"] > 2 * gm["InletGasT_std1h"]
                or prof.loc[s, "ConeDraft_std1h"] > 2 * gm["ConeDraft_std1h"]]
    ramp = [s for s in prof.index if s not in unstable and
            (prof.loc[s, "InletGasT"] < 750 or -prof.loc[s, "OutletDraft"] < 120)]
    stable = [s for s in prof.index if s not in unstable + ramp]
    for s in unstable:
        names[s] = "Unstable / disturbed (high variability, weak draft)"
    for s in ramp:
        names[s] = "Transition / low load (cool or weak draft)"
    if stable:
        hot = max(stable, key=lambda s: prof.loc[s, "InletGasT"])
        weak = max(stable, key=lambda s: prof.loc[s, "OutletDraft"])  # least negative = weakest suction
        for s in stable:
            if s == hot and len(stable) > 1:
                names[s] = "High load (hottest, strongest draft)"
            elif s == weak and len(stable) > 1:
                names[s] = "Reduced draft / lower gas flow"
            else:
                names[s] = "Normal operation"
    counts = pd.Series(names).value_counts()
    seen = {}
    for s in sorted(names):
        if counts[names[s]] > 1:
            seen[names[s]] = seen.get(names[s], 0) + 1
            names[s] = f"{names[s]} {seen[names[s]]}"
    for s, p in prof.iterrows():
        desc[s] = (f"Inlet gas {p['InletGasT']:.0f}°C, outlet gas {p['OutletGasT']:.0f}°C, material "
                   f"{p['MaterialT']:.0f}°C; inlet/cone/outlet draft {p['InletDraft']:.0f}/{p['ConeDraft']:.0f}/"
                   f"{p['OutletDraft']:.0f} mmWC; cyclone dP {p['DraftLoss']:.0f} mmWC; 1-h inlet-temp std "
                   f"{p['InletGasT_std1h']:.1f}°C, 1-h cone-draft std {p['ConeDraft_std1h']:.1f}")
    return names, desc


def cluster_states(clean, down, unknown, out, plots, summary):
    feats = build_state_features(clean)
    active_mask = (~down) & (~unknown) & feats[CLUSTER_FEATURES].notna().all(axis=1)
    X = feats.loc[active_mask, CLUSTER_FEATURES]
    log(f"Clustering {len(X):,} active rows on {len(CLUSTER_FEATURES)} features")

    # Robust scaling: clip extreme tails so a few spikes don't dominate centroids
    lo, hi = X.quantile(0.001), X.quantile(0.999)
    Xc = X.clip(lo, hi, axis=1)
    scaler = StandardScaler().fit(Xc)
    Z = scaler.transform(Xc)

    rng = np.random.default_rng(SEED)
    samp = rng.choice(len(Z), size=min(20000, len(Z)), replace=False)
    sel = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(Z)
        lab_s = km.labels_[samp]
        sel.append({"k": k, "inertia": km.inertia_,
                    "silhouette(20k sample)": silhouette_score(Z[samp], lab_s, random_state=SEED),
                    "davies_bouldin(20k sample)": davies_bouldin_score(Z[samp], lab_s),
                    "smallest_cluster_pct": 100 * np.bincount(km.labels_).min() / len(Z)})
    sel = pd.DataFrame(sel)
    sel.round(4).to_csv(out / "cluster_model_selection.csv", index=False)
    k_final = K_FINAL or choose_k(sel)
    log(f"Selected k={k_final}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].plot(sel.k, sel.inertia, "o-"); axes[0].set_title("Inertia (elbow)")
    axes[1].plot(sel.k, sel["silhouette(20k sample)"], "o-"); axes[1].set_title("Silhouette (higher better)")
    axes[2].plot(sel.k, sel["davies_bouldin(20k sample)"], "o-"); axes[2].set_title("Davies-Bouldin (lower better)")
    for ax in axes:
        ax.axvline(k_final, color="red", ls=":")
        ax.set_xlabel("k")
    savefig(fig, plots / "09_cluster_model_selection.png")

    km = KMeans(n_clusters=k_final, n_init=10, random_state=SEED).fit(Z)
    # Re-index clusters by descending mean inlet temperature for stable ids
    prof_raw = X.groupby(km.labels_).median()
    order = prof_raw["InletGasT_m1h"].sort_values(ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    raw_lab = pd.Series(-1, index=clean.index)
    raw_lab[active_mask] = pd.Series(km.labels_, index=X.index).map(remap)
    state = mode_filter(raw_lab, k_final, MODE_FILTER_POINTS)

    prof = feats.loc[state >= 0, PROFILE_FEATURES].groupby(state[state >= 0]).median()
    names, desc = name_states(prof)

    # Episodes (run-lengths) of each smoothed state
    ep_rows = []
    for s in range(k_final):
        r = runs(state == s)
        ep_rows.append({"state_id": s, "n_episodes": len(r),
                        "episode_mean_min": r.n.mean() * 5, "episode_median_min": r.n.median() * 5,
                        "episode_p90_min": r.n.quantile(0.9) * 5, "episode_max_h": r.n.max() * 5 / 60})
    ep = pd.DataFrame(ep_rows).set_index("state_id")

    # Transition matrix between consecutive episodes
    seq = state[state >= 0]
    chg = seq[seq.ne(seq.shift())]
    trans = pd.crosstab(chg.shift().dropna().astype(int), chg.iloc[1:].astype(int).values,
                        rownames=["from"], colnames=["to"])
    trans.to_csv(out / "state_transition_counts.csv")

    rows = []
    n_act = int((state >= 0).sum())
    for s in range(k_final):
        m = state == s
        r = {"state_id": s, "state_name": names[s], "description": desc[s],
             "n_points": int(m.sum()), "hours": round(m.sum() * 5 / 60, 1),
             "pct_of_active_time": round(100 * m.sum() / n_act, 2)}
        r.update(ep.loc[s].round(1).to_dict())
        for c in VARS:
            v = clean.loc[m, c]
            r[f"{SHORT[c]}_mean"] = round(v.mean(), 2)
            r[f"{SHORT[c]}_std"] = round(v.std(), 2)
            for p in (5, 25, 50, 75, 95):
                r[f"{SHORT[c]}_p{p:02d}"] = round(v.quantile(p / 100), 2)
        for c in ["DraftLoss", "InletGasT_std1h", "ConeDraft_std1h"]:
            r[f"{c}_median"] = round(feats.loc[m, c].median(), 2)
        rows.append(r)
    cs = pd.DataFrame(rows)

    # Scatter + time-share plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    samp_idx = X.sample(min(30000, len(X)), random_state=SEED).index
    for s in range(k_final):
        ii = samp_idx[state.loc[samp_idx].to_numpy() == s]
        axes[0].scatter(clean.loc[ii, TARGET], clean.loc[ii, "Cyclone_Outlet_Gas_draft"], s=2, alpha=0.4,
                        color=STATE_COLORS[s], label=f"S{s}: {names[s]}")
        axes[1].scatter(clean.loc[ii, "Cyclone_Material_Temp"], clean.loc[ii, "Cyclone_cone_draft"], s=2, alpha=0.4,
                        color=STATE_COLORS[s])
    axes[0].set_xlabel("Inlet gas temp (°C)"); axes[0].set_ylabel("Outlet gas draft (mmWC)")
    axes[1].set_xlabel("Material temp (°C)"); axes[1].set_ylabel("Cone draft (mmWC)")
    axes[0].legend(markerscale=6, fontsize=7, loc="lower left")
    axes[0].set_title("Operational states (30k-point sample)")
    axes[1].set_title("Same states: material temp vs cone draft")
    savefig(fig, plots / "10_cluster_scatter.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    share = pd.get_dummies(state[state >= 0]).resample("MS").mean() * 100
    share.columns = [f"S{c}: {names[c]}" for c in share.columns]
    share.plot.area(ax=ax, color=STATE_COLORS[:k_final], lw=0)
    ax.set_ylabel("% of active time")
    ax.set_title("Monthly share of operational states (active time only)")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    savefig(fig, plots / "11_state_share_monthly.png")

    # Profile heat-map (z of medians)
    fig, ax = plt.subplots(figsize=(11, 0.8 + 0.6 * k_final))
    act_prof = feats.loc[state >= 0, PROFILE_FEATURES]
    zprof = (prof - act_prof.median()) / act_prof.std()
    im = ax.imshow(zprof.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(len(PROFILE_FEATURES)), PROFILE_FEATURES, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(k_final), [f"S{s}: {names[s]}" for s in range(k_final)], fontsize=8)
    for i in range(k_final):
        for j in range(len(PROFILE_FEATURES)):
            ax.text(j, i, f"{prof.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="std-units vs active median")
    ax.set_title("State profiles (cell text = median value; colour = deviation from active median)")
    savefig(fig, plots / "12_state_profiles.png")

    summary["clustering"] = {
        "features": CLUSTER_FEATURES, "k_selected": int(k_final),
        "selection_table": sel.round(4).to_dict(orient="records"),
        "states": cs[["state_id", "state_name", "pct_of_active_time", "n_episodes", "episode_median_min",
                      "episode_p90_min", "description"]].to_dict(orient="records"),
    }
    return state, names, feats, cs, active_mask


def choose_k(sel: pd.DataFrame) -> int:
    """Pick k: best silhouette among k>=3 whose smallest cluster holds >= 2 % of data.
    k=2 is excluded because it only separates ramp/transition rows from everything else."""
    cand = sel[(sel.k >= 3) & (sel["smallest_cluster_pct"] >= 2.0)]
    if cand.empty:
        cand = sel[sel.k >= 3]
    return int(cand.sort_values("silhouette(20k sample)", ascending=False).iloc[0].k)


# --------------------------------------------------------------------------------------
# 4. Contextual anomaly detection
# --------------------------------------------------------------------------------------
ANOM_FEATURES = ["InletGasT", "OutletGasT", "MaterialT", "InletDraft", "ConeDraft", "OutletDraft",
                 "GasTempDrop", "DraftLoss", "ConeMinusInlet", "InletGasT_d15",
                 "InletGasT_std1h", "OutletDraft_std1h", "ConeDraft_std1h"]
SIGNATURE = {"GasTempDrop": "outlet-gas-temp", "OutletGasT": "outlet-gas-temp", "InletDraft": "draft",
             "OutletDraft": "draft", "DraftLoss": "draft", "ConeDraft": "cone-draft", "MaterialT": "material-temp",
             "InletGasT": "inlet-gas-temp"}
EXPLAIN_FEATURES = ["InletGasT", "OutletGasT", "MaterialT", "InletDraft", "ConeDraft", "OutletDraft",
                    "GasTempDrop", "DraftLoss"]


def detect_anomalies(clean, feats, state, names, down, sd, out, plots, summary):
    k = len(names)
    F = feats[ANOM_FEATURES]
    ok = (state >= 0) & F.notna().all(axis=1)
    score = pd.Series(np.nan, index=clean.index)
    zmax = pd.Series(np.nan, index=clean.index)
    zfull = pd.DataFrame(np.nan, index=clean.index, columns=EXPLAIN_FEATURES)
    thr = {}
    for s in range(k):
        m = ok & (state == s)
        Xs = F.loc[m]
        if len(Xs) < 500:
            continue
        sc = StandardScaler().fit(Xs)
        iso = IsolationForest(n_estimators=200, max_samples=1024, random_state=SEED, n_jobs=-1)
        iso.fit(sc.transform(Xs))
        sco = -iso.score_samples(sc.transform(Xs))  # higher = more anomalous
        score[m] = sco
        thr[s] = float(np.quantile(sco, IF_TOP_QUANTILE))
        med = Xs[EXPLAIN_FEATURES].median()
        mad = (Xs[EXPLAIN_FEATURES] - med).abs().median() * 1.4826
        z = (Xs[EXPLAIN_FEATURES] - med) / mad.replace(0, np.nan)
        zfull.loc[m] = z.values
        zmax[m] = z.abs().max(axis=1)

    thr_s = state.map(thr)
    path_if = (score > thr_s) & (zmax > ROBUST_Z_MIN)
    path_z = zmax > ROBUST_Z_STRONG
    flag = path_if | path_z
    summary["anomaly_paths"] = {"points_iforest_and_z5": int(path_if.sum()), "points_z10_only": int((path_z & ~path_if).sum())}

    # Group into events (merge flagged points separated by small gaps)
    f = flag.to_numpy().copy()
    gaps = runs(pd.Series(~f, index=flag.index))
    for r in gaps.itertuples():
        if r.n <= EVENT_MERGE_POINTS - 1 and r.i0 > 0 and r.i1 < len(f) and f[r.i0 - 1] and f[r.i1]:
            f[r.i0: r.i1] = True
    ev = runs(pd.Series(f, index=flag.index))

    sd_starts = sd["start"].to_numpy()
    sd_ends = sd["end"].to_numpy()
    rows = []
    for i, r in enumerate(ev.itertuples(), 1):
        idx = clean.index[r.i0: r.i1]
        z = zfull.loc[idx]
        zmean = z.mean()
        top = zmean.abs().sort_values(ascending=False).head(3)
        st = state.loc[idx]
        st = st[st >= 0]
        dom = int(st.mode().iloc[0]) if len(st) else -1
        start, end = idx[0], idx[-1] + pd.Timedelta(FREQ)
        nxt = sd_starts[sd_starts >= np.datetime64(end)]
        prv = sd_ends[sd_ends <= np.datetime64(start)]
        to_next = (nxt[0] - np.datetime64(end)) / np.timedelta64(1, "m") if len(nxt) else np.nan
        since_prev = (np.datetime64(start) - prv[-1]) / np.timedelta64(1, "m") if len(prv) else np.nan
        prev_state = state.loc[:start].iloc[:-1]
        prev_state = prev_state[prev_state >= 0]
        rows.append({
            "event_id": i, "start": start, "end": end, "duration_minutes": int(r.n * 5),
            "n_flagged_points": int(flag.loc[idx].sum()),
            "state_id": dom, "state_name": names.get(dom, "n/a"),
            "peak_iforest_score": round(float(score.loc[idx].max()), 4),
            "state_threshold": round(thr.get(dom, np.nan), 4),
            "max_robust_z": round(float(zmax.loc[idx].max()), 2),
            "top_variables": "; ".join(f"{v} ({'+' if zmean[v] > 0 else '-'}{abs(zmean[v]):.1f} rz)" for v in top.index),
            "primary_variable": top.index[0],
            "signature": SIGNATURE.get(top.index[0], top.index[0]),
            "severity": round(float(((zmax.loc[idx].clip(upper=50) / ROBUST_Z_MIN)
                                     * (1 + ((score.loc[idx] - thr_s.loc[idx]) / thr_s.loc[idx]).clip(lower=0))).sum()), 2),
            "detection_path": "iforest+z" if path_if.loc[idx].any() else "strong-z only",
            "minutes_to_next_shutdown": round(to_next, 0),
            "minutes_since_prev_shutdown": round(since_prev, 0),
            "state_before": names.get(int(prev_state.iloc[-1]), "n/a") if len(prev_state) else "n/a",
            "context": ("pre-shutdown (<=2 h before stop)" if pd.notna(to_next) and to_next <= 120 else
                        "post-restart (<=6 h after restart)" if pd.notna(since_prev) and since_prev <= 360 else
                        "steady operation"),
            **{f"mean_{SHORT[c]}": round(float(clean.loc[idx, c].mean()), 1) for c in VARS},
        })
    an = pd.DataFrame(rows)
    an.to_csv(out / "anomalous_periods.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    # Per-state anomaly statistics
    per_state = []
    for s in range(k):
        m = ok & (state == s)
        e = an[an.state_id == s]
        hrs = m.sum() * 5 / 60
        per_state.append({"state_id": s, "state_name": names[s], "flagged_point_rate_pct": round(100 * flag[m].mean(), 3),
                          "n_events": int(len(e)), "events_per_1000_active_h": round(1000 * len(e) / max(hrs, 1), 2)})
    per_state = pd.DataFrame(per_state)

    pre_sd = an["minutes_to_next_shutdown"].le(120)
    # Baseline: fraction of active time that lies within 2 h before a shutdown
    pre_mask = pd.Series(False, index=clean.index)
    for e in sd.itertuples():
        pre_mask.loc[e.start - pd.Timedelta("2h"): e.start - pd.Timedelta(FREQ)] = True
    base_frac = float(pre_mask[ok].mean())
    s = {
        "method": f"per-state IsolationForest (top {100 * (1 - IF_TOP_QUANTILE):.0f} % score) AND state-conditional "
                  f"robust |z| > {ROBUST_Z_MIN}; events merged across gaps <= {EVENT_MERGE_POINTS * 5 - 5} min",
        "n_flagged_points": int(flag.sum()), "flagged_pct_of_active": round(100 * flag[ok].mean(), 3),
        "n_events": int(len(an)),
        "event_duration_median_min": float(an.duration_minutes.median()) if len(an) else 0,
        "events_ge_30min": int((an.duration_minutes >= 30).sum()) if len(an) else 0,
        "primary_variable_counts": an.primary_variable.value_counts().to_dict() if len(an) else {},
        "signature_counts": an.signature.value_counts().to_dict() if len(an) else {},
        "outlet_temp_signature_events_by_year": an[an.signature == "outlet-gas-temp"].groupby(an.start.dt.year).size().to_dict() if len(an) else {},
        "outlet_temp_signature_events_mar_jun_2018": int(((an.signature == "outlet-gas-temp") & (an.start >= "2018-03-01") & (an.start < "2018-07-01")).sum()) if len(an) else 0,
        "per_state": per_state.to_dict(orient="records"),
        "events_by_context": an.context.value_counts().to_dict() if len(an) else {},
        "events_by_detection_path": an.detection_path.value_counts().to_dict() if len(an) else {},
        "pct_events_within_2h_before_shutdown": round(100 * pre_sd.mean(), 1) if len(an) else 0,
        "baseline_pct_active_time_within_2h_before_shutdown": round(100 * base_frac, 1),
        "pct_shutdowns_preceded_by_anomaly_within_2h": round(100 * np.mean([
            ((an.end <= e.start) & (an.end > e.start - pd.Timedelta("2h"))).any() for e in sd.itertuples()]), 1) if len(sd) else 0,
    }
    summary["anomalies"] = s
    log(f"Anomalies: {s['n_events']} events, {s['flagged_pct_of_active']} % of active points")

    # Overview plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].scatter(an.start, an.duration_minutes, s=6 + 30 * an.severity / max(an.severity.max(), 1e-9),
                    c=[STATE_COLORS[max(s_, 0)] for s_ in an.state_id], alpha=0.6)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Event duration (min)")
    axes[0].set_title("Anomalous events over time (colour = operational state, size = severity)")
    mon = an.groupby([an.start.dt.to_period("M"), "signature"]).size().unstack(fill_value=0)
    mon = mon.reindex(pd.period_range(an.start.min(), an.start.max(), freq="M"), fill_value=0)
    bottom = np.zeros(len(mon))
    for j, c in enumerate(mon.columns):
        axes[1].bar(mon.index.to_timestamp(), mon[c].values, width=25, bottom=bottom, label=c,
                    color=STATE_COLORS[j % len(STATE_COLORS)])
        bottom += mon[c].values
    axes[1].legend(fontsize=8, ncol=5); axes[1].set_ylabel("events / month")
    axes[1].set_title("Anomalous events per month by signature (most-implicated variable group)")
    savefig(fig, plots / "13_anomaly_overview.png")

    cases = pick_case_studies(an)
    case_rows = []
    for j, e in enumerate(cases.itertuples(), 1):
        ev_info = case_study(j, e, clean, feats, state, names, score, thr_s, flag, sd, plots, zfull)
        case_rows.append(ev_info)
    cs = pd.DataFrame(case_rows)
    cs.to_csv(out / "anomaly_case_studies.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    summary["anomaly_case_studies"] = cs.to_dict(orient="records")
    return an, flag, score, per_state


def pick_case_studies(an: pd.DataFrame) -> pd.DataFrame:
    """Deterministic selection so the case studies cover *different* failure signatures:
    events >= 15 min, ranked by severity; one per signature; >= 7 days apart; at most one post-restart ramp;
    the strongest pre-shutdown event (with a not-yet-used signature) is forced in as a precursor example."""
    c = an[an.duration_minutes >= 15].sort_values("severity", ascending=False)
    chosen = []

    def ok(e, n_restart_max=1):
        if e.signature in {x.signature for x in chosen}:
            return False
        if any(abs((e.start - x.start).total_seconds()) < 7 * 86400 for x in chosen):
            return False
        if e.context.startswith("post-restart") and sum(x.context.startswith("post-restart") for x in chosen) >= n_restart_max:
            return False
        return True

    it = list(c.itertuples())
    if it:
        chosen.append(it[0])
    for e in it:
        if e.context.startswith("pre-shutdown") and ok(e):
            chosen.append(e)
            break
    for e in it:
        if len(chosen) >= N_CASE_STUDIES:
            break
        if ok(e):
            chosen.append(e)
    out = pd.DataFrame(chosen).drop(columns="Index", errors="ignore")
    return out.sort_values("start").reset_index(drop=True)


def case_study(j, e, clean, feats, state, names, score, thr_s, flag, sd, plots, zfull):
    pre_all = clean.loc[e.start - pd.Timedelta("2h"): e.start - pd.Timedelta(FREQ)]
    pre = pre_all[state.loc[pre_all.index] >= 0]          # active points only (exclude shutdown ramps)
    dur = clean.loc[e.start: e.end - pd.Timedelta(FREQ)]
    delta = (dur.mean() - pre.mean()) if len(pre) >= 6 else pd.Series(np.nan, index=VARS)
    rz = zfull.loc[dur.index].mean()                        # deviation vs the state's own median (contextual)
    # Which variable moved first? first time its 5-min change exceeds 4x its pre-window std of changes
    w = clean.loc[e.start - pd.Timedelta("1h"): e.end]
    first = {}
    for c in VARS:
        dd = w[c].diff()
        sd_ = pre[c].diff().std()
        hit = dd.abs() > 4 * (sd_ if sd_ and sd_ > 0 else 1)
        first[SHORT[c]] = w.index[hit.to_numpy()][0] if hit.any() else pd.NaT
    order = sorted([(t, v) for v, t in first.items() if pd.notna(t)])
    st_before = state.loc[: e.start].iloc[-13:-1]
    st_after = state.loc[e.end:].iloc[1:13]
    sd_near = sd[(sd.start >= e.start - pd.Timedelta("6h")) & (sd.start <= e.end + pd.Timedelta("6h"))]

    info = {
        "case": j, "event_id": e.event_id, "start": e.start, "end": e.end, "duration_minutes": e.duration_minutes,
        "state_name": e.state_name, "context": e.context, "top_variables": e.top_variables,
        **{f"rz_vs_state_{v}": round(float(rz[v]), 1) for v in EXPLAIN_FEATURES},
        **{f"change_vs_prev2h_active_{SHORT[c]}": round(float(delta[c]), 1) for c in VARS},
        "first_movers": " -> ".join(f"{v}@{t:%H:%M}" for t, v in order[:4]) if order else "none",
        "state_before": names.get(int(st_before.mode().iloc[0]), "shutdown/unknown") if (st_before >= 0).any() else "shutdown/unknown",
        "state_after": names.get(int(st_after.mode().iloc[0]), "shutdown/unknown") if len(st_after) and (st_after >= 0).any() else "shutdown/unknown",
        "shutdown_within_6h": bool(len(sd_near)),
        "minutes_to_next_shutdown": e.minutes_to_next_shutdown,
        "auto_hypothesis": auto_hypothesis(rz, e.context),
    }

    # Plot ±6 h context
    t0, t1 = e.start - pd.Timedelta("6h"), e.end + pd.Timedelta("6h")
    w = clean.loc[t0:t1]
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 0.6, 1.5]})
    for c in TEMPS:
        axes[0].plot(w.index, w[c], lw=1, label=SHORT[c])
    for c in DRAFTS:
        axes[1].plot(w.index, w[c], lw=1, label=SHORT[c])
    axes[0].set_ylabel("°C"); axes[1].set_ylabel("mmWC")
    axes[0].legend(fontsize=8, ncol=3); axes[1].legend(fontsize=8, ncol=3)
    stw = state.loc[t0:t1]
    for s in sorted(set(stw.unique())):
        mm = stw == s
        col = "#999999" if s < 0 else STATE_COLORS[s]
        axes[2].fill_between(stw.index, 0, 1, where=mm.to_numpy(), color=col, step="mid",
                             label="shutdown/unknown" if s < 0 else f"S{s}: {names[s]}")
    axes[2].set_yticks([]); axes[2].legend(fontsize=7, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    axes[2].set_ylabel("state", fontsize=8)
    axes[3].plot(w.index, score.loc[t0:t1], color="k", lw=0.8, label="IF score")
    axes[3].plot(w.index, thr_s.loc[t0:t1], color="r", ls=":", lw=1, label="state threshold")
    axes[3].legend(fontsize=8)
    for ax in (axes[0], axes[1], axes[3]):
        ax.axvspan(e.start, e.end, color="red", alpha=0.15)
        for s_ in sd_near.itertuples():
            ax.axvspan(max(s_.start, t0), min(s_.end, t1), color="grey", alpha=0.2)
    axes[0].set_title(f"Case {j} (event #{e.event_id}): {e.start:%Y-%m-%d %H:%M}, {e.duration_minutes} min, "
                      f"state = {e.state_name}\nTop deviations: {e.top_variables}", fontsize=10)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    fig.text(0.01, 0.005, "Red band = anomalous event; grey band = detected shutdown. Hypothesis (unconfirmed): "
             + info["auto_hypothesis"][:160], fontsize=7.5)
    savefig(fig, plots / f"14_anomaly_case_{j}.png")
    return info


def auto_hypothesis(rz: pd.Series, context: str) -> str:
    """Heuristic *hypothesis* from state-relative robust z-scores (rz). Not a confirmed cause."""
    g = lambda k: float(rz.get(k, 0.0)) if pd.notna(rz.get(k, np.nan)) else 0.0
    msgs = []
    if g("OutletGasT") < -5 and abs(g("InletGasT")) < 3 and abs(g("MaterialT")) < 3:
        msgs.append("outlet gas temperature far below inlet while inlet and material temperatures stay normal -> "
                    "physically implausible heat loss; most likely outlet thermocouple fault/fouling or local false-air "
                    "ingress at the measuring point")
    if g("MaterialT") < -5 and abs(g("InletGasT")) < 3:
        msgs.append("material temperature collapsed while gas temperatures stayed normal -> material-temperature "
                    "sensor fault or a flush of cold/unprocessed material")
    if g("MaterialT") > 5 and abs(g("InletGasT")) < 3:
        msgs.append("material temperature high on its own -> low feed rate (less material absorbing heat) or sensor drift")
    if g("ConeDraft") > 4 and abs(g("OutletDraft")) < 2:
        msgs.append("cone draft moved toward atmospheric while outlet draft held -> possible material build-up / "
                    "partial cone or dip-tube blockage")
    weak = [g(k) for k in ("InletDraft", "ConeDraft", "OutletDraft")]
    if min(weak) > 3:
        tail = (" - consistent with the onset of the following stop" if context.startswith("pre-shutdown") else
                " - draft not yet restored after restart (fan/damper ramp)" if context.startswith("post-restart") else "")
        msgs.append("all drafts weaker than normal for this state -> reduced ID-fan suction / damper closing / lower "
                    "gas flow" + tail)
    if max(weak) < -3:
        msgs.append("all drafts stronger than normal -> fan speed increase / damper opening or upstream draft surge")
    if g("InletGasT") > 4:
        msgs.append("inlet gas hotter than normal -> upstream firing surge or reduced feed (less heat sink)")
    if g("InletGasT") < -4:
        msgs.append("inlet gas cooler than normal -> reduced firing, false-air ingress or the start of a stop")
    if not msgs:
        msgs.append("moderate multivariate deviation with no single dominant driver -> transient process disturbance; "
                    "cross-check with operator log")
    return "; ".join(msgs)


# --------------------------------------------------------------------------------------
# 5. Forecasting
# --------------------------------------------------------------------------------------
def make_forecast_features(clean: pd.DataFrame) -> pd.DataFrame:
    y = clean[TARGET]
    f = pd.DataFrame(index=clean.index)
    for l in range(0, 13):
        f[f"y_lag{l}"] = y.shift(l)
    for l in (18, 24, 36, 72, 288):
        f[f"y_lag{l}"] = y.shift(l)
    for w in (3, 6, 12, 36):
        f[f"y_mean{w}"] = y.rolling(w).mean()
        f[f"y_std{w}"] = y.rolling(w).std()
    f["y_d1"] = y.diff(1)
    f["y_d3"] = y.diff(3)
    f["y_d12"] = y.diff(12)
    for c in VARS[1:]:
        s = clean[c]
        f[f"{SHORT[c]}_0"] = s
        f[f"{SHORT[c]}_d1"] = s.diff(1)
        f[f"{SHORT[c]}_d6"] = s.diff(6)
        f[f"{SHORT[c]}_mean12"] = s.rolling(12).mean()
    med_draft = clean[DRAFTS].median(axis=1)
    f["draft_off_now"] = (med_draft > DRAFT_OFF).astype(float)          # causal instantaneous state
    f["temp_off_now"] = (y < TEMP_OFF).astype(float)
    f["hour_sin"] = np.sin(2 * np.pi * clean.index.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * clean.index.hour / 24)
    return f


def forecast(clean, down, out, plots, summary):
    y = clean[TARGET]
    X = make_forecast_features(clean)
    Y = pd.concat({h: y.shift(-h) for h in range(1, HORIZON + 1)}, axis=1)
    valid = X.notna().all(axis=1) & Y.notna().all(axis=1)
    test_start = pd.Timestamp(TEST_START)
    tr = valid & (X.index < test_start - pd.Timedelta(minutes=5 * HORIZON))  # purge: no target overlap into test
    te = valid & (X.index >= test_start) & (X.index.minute == 0)            # non-overlapping hourly origins
    tr_idx = X.index[tr][::TRAIN_STRIDE]
    te_idx = X.index[te]
    log(f"Forecast: train origins {len(tr_idx):,} (stride {TRAIN_STRIDE}), test hourly origins {len(te_idx):,}")

    y0_tr = X.loc[tr_idx, "y_lag0"].to_numpy()
    y0_te = X.loc[te_idx, "y_lag0"].to_numpy()
    preds = {"persistence": np.repeat(y0_te[:, None], HORIZON, axis=1)}

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(X.loc[tr_idx], Y.loc[tr_idx].to_numpy() - y0_tr[:, None])  # predict change vs. now
    preds["ridge_ar"] = y0_te[:, None] + ridge.predict(X.loc[te_idx])

    hgb_pred = np.zeros((len(te_idx), HORIZON))
    for h in range(1, HORIZON + 1):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=63,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          loss="squared_error", random_state=SEED, early_stopping=False)
        m.fit(X.loc[tr_idx], Y.loc[tr_idx, h].to_numpy() - y0_tr)
        hgb_pred[:, h - 1] = y0_te + m.predict(X.loc[te_idx])
    preds["hist_gbm"] = hgb_pred

    actual = Y.loc[te_idx].to_numpy()
    # Regime of each test origin: "steady" = no shutdown/transition in [t-1h, t+1h] and inlet temp >= 750 throughout
    dn = down.astype(int).rolling(25, center=True, min_periods=1).max()
    lowT = (y < 750).astype(int).rolling(25, center=True, min_periods=1).max()
    regime = np.where((dn.loc[te_idx] == 0) & (lowT.loc[te_idx] == 0), "steady_operation", "shutdown_or_transition")

    rows = []
    for name, p in preds.items():
        err = p - actual
        for reg in ["all", "steady_operation", "shutdown_or_transition"]:
            mm = np.ones(len(te_idx), bool) if reg == "all" else (regime == reg)
            if mm.sum() == 0:
                continue
            e_ = err[mm]
            rows.append({"model": name, "subset": reg, "n_origins": int(mm.sum()), "horizon": "1-12 (all)",
                         "RMSE": float(np.sqrt(np.mean(e_ ** 2))), "MAE": float(np.mean(np.abs(e_)))})
            for h in range(HORIZON):
                rows.append({"model": name, "subset": reg, "n_origins": int(mm.sum()), "horizon": str(h + 1),
                             "RMSE": float(np.sqrt(np.mean(e_[:, h] ** 2))), "MAE": float(np.mean(np.abs(e_[:, h])))})
    met = pd.DataFrame(rows)
    met[["RMSE", "MAE"]] = met[["RMSE", "MAE"]].round(3)
    met.to_csv(out / "forecast_metrics.csv", index=False)

    # forecasts.csv (long format; reproduces forecast_metrics.csv exactly)
    long = []
    for h in range(HORIZON):
        long.append(pd.DataFrame({
            "origin_time": te_idx, "target_time": te_idx + pd.Timedelta(minutes=5 * (h + 1)),
            "horizon_step": h + 1, "horizon_minutes": 5 * (h + 1), "regime": regime,
            "actual": actual[:, h].round(3),
            **{f"pred_{k}": v[:, h].round(3) for k, v in preds.items()},
        }))
    fc = pd.concat(long).sort_values(["origin_time", "horizon_step"])
    fc.to_csv(out / "forecasts.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")

    # Plots: metrics by horizon
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for col, reg in enumerate(["all", "steady_operation"]):
        for name in preds:
            mm = met[(met.model == name) & (met.subset == reg) & (met.horizon != "1-12 (all)")]
            axes[0, col].plot(mm.horizon.astype(int) * 5, mm.RMSE, "o-", label=name)
            axes[1, col].plot(mm.horizon.astype(int) * 5, mm.MAE, "o-", label=name)
        axes[0, col].set_title(f"RMSE by horizon - {reg}")
        axes[1, col].set_title(f"MAE by horizon - {reg}")
        axes[1, col].set_xlabel("Horizon (minutes)")
        axes[0, col].set_ylabel("°C"); axes[1, col].set_ylabel("°C")
        axes[0, col].legend(fontsize=8)
    savefig(fig, plots / "15_forecast_error_by_horizon.png")

    # Example forecasts: continuous rolling 1-h-ahead (h=12) over a 10-day test window, plus 4 trajectories
    win = fc[(fc.horizon_step == 12)].set_index("target_time")
    sl = win.loc["2020-02-01":"2020-02-10"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(sl.index, sl.actual, color="k", lw=1.2, label="actual")
    for name in preds:
        axes[0].plot(sl.index, sl[f"pred_{name}"], lw=0.9, label=f"{name} (60-min-ahead)")
    axes[0].legend(fontsize=8); axes[0].set_ylabel("°C")
    axes[0].set_title("Test period: 60-minute-ahead forecasts issued hourly (Feb 2020)")
    trans_orig = fc[(fc.regime == "shutdown_or_transition") & (fc.horizon_step == 12)]
    trans_orig = trans_orig.assign(err=(trans_orig.actual - trans_orig.pred_persistence).abs()).sort_values("err").origin_time
    picks = list(trans_orig.iloc[-1:]) + list(fc[fc.regime == "steady_operation"].origin_time.drop_duplicates().iloc[[100, 1000]])
    for o in picks:
        traj = fc[fc.origin_time == o]
        hist = y.loc[o - pd.Timedelta("2h"): o]
        l, = axes[1].plot(hist.index, hist.values, color="grey", lw=1)
        axes[1].plot(traj.target_time, traj.actual, "k.-", lw=1)
        for name, ls in zip(preds, ["--", "-.", "-"]):
            axes[1].plot(traj.target_time, traj[f"pred_{name}"], ls=ls, lw=1.2, label=f"{name}" if o == picks[0] else None)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Example 12-step trajectories (grey = history, black = actual) - incl. the worst transition origin")
    savefig(fig, plots / "16_forecast_examples.png")

    overall = met[met.horizon == "1-12 (all)"].set_index(["model", "subset"])[["RMSE", "MAE", "n_origins"]]
    h12 = met[met.horizon == "12"].set_index(["model", "subset"])[["RMSE", "MAE"]]
    summary["forecasting"] = {
        "split": f"train 2017-01-01 .. {TEST_START} minus 1 h purge; test {TEST_START} .. end (hourly origins)",
        "n_train_origins": int(len(tr_idx)), "n_test_origins": int(len(te_idx)),
        "overall": {f"{a}|{b}": v for (a, b), v in overall.round(3).to_dict(orient="index").items()},
        "h12": {f"{a}|{b}": v for (a, b), v in h12.round(3).to_dict(orient="index").items()},
    }
    log("Forecast metrics (all horizons):\n" + overall.round(2).to_string())


# --------------------------------------------------------------------------------------
def insights_numbers(sd, an, state, names, down, clean, summary):
    """Extra cross-analysis numbers used in the README insights."""
    # Anomaly rate per state normalised by active hours already in summary; here: post-start-up behaviour
    sd_ = sd.copy()
    sd_["next_gap_h"] = (sd_.start.shift(-1) - sd_.end).dt.total_seconds() / 3600
    summary["cross"] = {
        "median_run_between_shutdowns_h": round(float(sd_.next_gap_h.median()), 1),
        "pct_restarts_followed_by_another_stop_within_24h": round(100 * float((sd_.next_gap_h < 24).mean()), 1),
        "anomaly_events_within_6h_after_restart_pct": round(100 * float(an.minutes_since_prev_shutdown.le(360).mean()), 1),
        "baseline_active_time_within_6h_after_restart_pct": None,
    }
    post = pd.Series(False, index=clean.index)
    for e in sd.itertuples():
        post.loc[e.end: e.end + pd.Timedelta("6h")] = True
    act = state >= 0
    summary["cross"]["baseline_active_time_within_6h_after_restart_pct"] = round(100 * float(post[act].mean()), 1)


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--data", default=str(here / "data" / "data.xlsx"))
    ap.add_argument("--out", default=str(here))
    args = ap.parse_args()
    out = Path(args.out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (out / "outputs").mkdir(exist_ok=True)
    t0 = time.time()
    np.random.seed(SEED)
    summary: dict = {}

    aux = out / "outputs"  # supporting outputs; the 4 required CSVs go to the Task1 root
    clean = load_and_clean(Path(args.data), aux, plots, summary)
    down, unknown, sd = detect_shutdowns(clean, out, plots, summary)
    eda(clean, down | unknown, aux, plots, summary)
    state, names, feats, cs, active_mask = cluster_states(clean, down, unknown, aux, plots, summary)
    an, flag, score, per_state = detect_anomalies(clean, feats, state, names, down, sd, aux, plots, summary)
    cs = cs.merge(per_state[["state_id", "flagged_point_rate_pct", "n_events", "events_per_1000_active_h"]]
                  .rename(columns={"n_events": "n_anomaly_events", "flagged_point_rate_pct": "anomalous_point_rate_pct",
                                   "events_per_1000_active_h": "anomaly_events_per_1000_h"}), on="state_id")
    cs.to_csv(out / "clusters_summary.csv", index=False)
    # Move required CSV for anomalies to root as well
    (aux / "anomalous_periods.csv").replace(out / "anomalous_periods.csv")
    insights_numbers(sd, an, state, names, down, clean, summary)
    forecast(clean, down, out, plots, summary)
    (out / "forecast_metrics.csv").replace(aux / "forecast_metrics.csv")

    # Per-timestamp labels (handy for downstream use)
    lab = pd.DataFrame({"is_shutdown": down, "state_unknown_data_gap": unknown, "state_id": state,
                        "anomaly_flag": flag, "anomaly_score": score.round(4)})
    lab["state_name"] = lab.state_id.map(names).fillna("shutdown/unknown")
    lab.to_csv(aux / "timestamp_labels.csv.gz", date_format="%Y-%m-%d %H:%M:%S", compression="gzip")

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    with open(aux / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Done in {summary['runtime_seconds']} s. Outputs in {out}")


if __name__ == "__main__":
    main()
