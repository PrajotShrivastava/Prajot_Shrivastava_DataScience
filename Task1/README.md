# Task 1: cyclone machine time-series analysis

This is a reproducible end-to-end analysis of the cyclone sensor data in `data/data.xlsx`. It covers data quality, shutdown detection, operating-state clustering, state-contextual anomaly detection with root-cause hypotheses, 1-hour forecasting, and operational insights.
**Every number below is produced by `task1_analysis.py`**, and most are also stored in `outputs/results_summary.json`.

## How to run

```bash
cd Task1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python task1_analysis.py                       # uses data/data.xlsx; ~1.5 min on a laptop CPU
python task1_analysis.py --data /path/to/data.xlsx --out /some/output/dir    # optional
```

This was tested with Python 3.14 on macOS arm64 (exact versions are in `requirements.txt`). The run is deterministic: `SEED = 42`, and all thresholds are constants at the top of the script. The console log of the final run is in `outputs/run_log.txt`.

## Deliverables

| File | Content |
|---|---|
| `task1_analysis.py` | The full pipeline, split into 5 numbered sections that map to the task list |
| `shutdown_periods.csv` | 154 detected shutdown / idle events: start, end (exclusive), duration, minimum inlet temperature, which rule fired, data-gap flag, type |
| `anomalous_periods.csv` | 746 anomalous events: start, end, duration, state, IF score vs. state threshold, max robust z, **top implicated variables**, failure signature, detection path, context (pre-stop / post-restart / steady), mean sensor values |
| `clusters_summary.csv` | One row per operating state: name, description, share of time, episode statistics, mean / std / p05–p95 of all 6 sensors, anomaly rate |
| `forecasts.csv` | 62,316 rows = 5,193 hourly test origins × 12 horizons: actual value and predictions from persistence, Ridge AR-X and HistGradientBoosting. `outputs/forecast_metrics.csv` can be recomputed exactly from this file. |
| `plots/*.png` | 22 figures (list at the end) |
| `outputs/` | Supporting outputs: `data_quality_report.json`, `summary_statistics.csv`, `correlation_matrix_{all,active}.csv`, `cluster_model_selection.csv`, `state_transition_counts.csv`, `anomaly_case_studies.csv`, `forecast_metrics.csv`, `timestamp_labels.csv.gz` (per-timestamp shutdown flag, state, anomaly flag and score), `results_summary.json`, `run_log.txt` |
| `data/data.xlsx` | Copy of the provided dataset (the original stays untouched outside the submission) |

---

## 1. Data preparation and EDA

| Check | Finding | Treatment |
|---|---|---|
| Span | 377,719 rows, **2017-01-01 00:00 → 2020-08-07 12:15**. That is about 3.6 years, more than the "3 years" stated in the brief. | Kept the full range |
| Timestamps | All parse correctly. No duplicates, nothing off the 5-minute grid, monotonic. | Reindexed to a strict 5-minute grid of 378,580 slots |
| Gaps | The only gaps are **31 December of 2017, 2018 and 2019** (287 slots each, 861 in total) | Inserted as NaN. The state is "unknown", not counted as downtime. |
| Historian status text | 8,195 cells such as `Not Connect`, `I/O Timeout`, `Configure`, `Scan Timeout`, `Unit Down`, `Comm Fail` | → NaN |
| Out of physical range | Negative material temperatures, temperatures pinned at 1375 °C (instrument ceiling), one +488 mmWC cone-draft spike (118 cells) | → NaN |
| Cross-sensor implausibility | Material temperature more than 250 °C below inlet gas temperature while gas > 750 °C and the fan is running (1,493 points). Clustering at k ≥ 5 had isolated these as their own "state". | Material temperature → NaN |
| Stuck values | The same value repeated for ≥ 3 h, mostly material temperature during shutdowns (28,610 points) | → NaN |
| Statistical spikes | Hampel z > 10 is counted per sensor in the report | **Flagged, not removed.** Extremes can be real process events and are left for the anomaly detector. |
| Short gaps | | Time interpolation for gaps ≤ 15 min only. After cleaning, 344,448 rows have all 6 sensors. |

- **Summary statistics** are in `outputs/summary_statistics.csv`, for all data and for active operation only.
- **Correlations** are in `plots/03_correlation_matrix.png`.
  - Across all data, everything correlates at |r| ≈ 0.9, because the on/off switching dominates.
  - Within active operation, the three drafts move together (inlet vs outlet r = 0.96).
  - Inlet vs outlet gas temperature is only r = 0.36 during operation. That low value is a clue to the 2018 outlet-thermocouple problem described below.
- **Plots:**
  - `04_week_slice.png` shows a representative running week (2018-07-16).
  - `05_year_slice_2018.png` shows daily median and 5–95 % bands for 2018.
  - `02_distributions.png` shows the bimodal (running vs stopped) distributions.

## 2. Shutdown / idle detection

**Rule:** a point counts as "down" if **the median of the 3 drafts is above −60 mmWC** (ID fan off / no gas flow) **or the inlet gas temperature is below 300 °C**.
- Down runs separated by less than 30 min of running are merged.
- Down runs shorter than 30 min are discarded; they are dips, not stops.
- Sensor gaps of up to 1 h carry the previous state forward.

**Why this rule:**
- Draft is the cleanest on/off signal. It is strongly bimodal: about −150 to −300 mmWC when running, about 0 to +20 mmWC when stopped, with a sparse valley in between (−60 sits inside it).
- During a stop, draft collapses within minutes, but the gas stays at 700–800 °C for hours while it cools (`08_shutdown_rule_example.png`). A temperature-only threshold would count those cooling hours as running.
- The temperature condition catches cold periods where a draft sensor is missing.
- The median of the three drafts is robust to one faulty draft sensor.

**Results:**
- **154 events and 6,840 h of downtime (285 days, 21.7 % of the record).**
  - 73 short stops (< 2 h), 48 stops of 2–24 h, and 33 extended stops (> 24 h). The extended stops account for about 95 % of the downtime.
  - The longest stops are 2019-09-14 → 2019-10-26 (1,010 h) and 2020-03-22 → 2020-05-02 (964 h).
- **By year:**

  | Year | Events | Downtime |
  |---|---|---|
  | 2017 | 49 | 1,478 h |
  | 2018 | 56 | 1,387 h |
  | 2019 | 38 | 2,958 h |
  | 2020 (partial) | 11 | 1,017 h |

- **Plots:** `06_shutdowns_year_2018.png` (headline full-year figure; 2017 and 2019 versions are also provided) and `07_downtime_monthly_and_durations.png`.
- **Caveat:** without an operator log I cannot separate planned from unplanned stops. Very long stops are probably planned (maintenance or market-driven).

## 3. Operating-state segmentation

- **Data:** active points only. Shutdowns and unknown periods are excluded, leaving 292k points.
- **Features (10):** 1-hour trailing means of the 6 sensors and of the cyclone pressure drop (inlet minus outlet draft), plus 1-hour standard deviations of inlet temperature, outlet draft and cone draft.
  - The rolling means describe operating *regimes* rather than 5-minute noise.
  - The standard deviations capture stability.
  - Extremes are clipped at 0.1 / 99.9 %, then standardised.
- **Model selection:** KMeans was run for k = 2…8 (`cluster_model_selection.csv`, `09_cluster_model_selection.png`).

  | k | Silhouette | What it shows |
  |---|---|---|
  | 2–3 | ≈ 0.24 (best) | Merges two strong-draft regimes that differ by about 20 °C in inlet temperature and in gas cooling |
  | 4 (chosen) | 0.18 | Separates those regimes; the smallest cluster still holds ≥ 2 % of points |
  | ≥ 5 | | Splits off tiny clusters that turned out to be sensor faults (now removed in cleaning) |

  Silhouette is low for every k. The data is a continuum, not well-separated blobs, so the states are useful *descriptions* of operating regimes, not sharply distinct modes. HDBSCAN/DBSCAN were not used: density methods on 292k points of continuous, heavily overlapping data tend to produce one giant cluster plus noise, and KMeans centroids are easier to name.
- **Smoothing:** a 35-minute centred majority filter removes label flicker. Median episodes are 65–108 min.

| State | % of active time | Median episode | Profile (medians) |
|---|---|---|---|
| S0 High load (hottest, strongest draft) | 36.1 % | 100 min | inlet 902 °C, outlet draft −256, ΔP 54 |
| S1 Unstable / disturbed (high variability, weak draft) | 2.4 % | 65 min | 1-h inlet-temperature std 49 °C (normal ≈ 12–18), outlet draft −156, cone-draft std 31 |
| S2 Reduced draft / lower gas flow | 33.6 % | 108 min | inlet 883 °C, outlet draft −192, ΔP 36 |
| S3 Normal operation | 27.9 % | 85 min | inlet 883 °C, outlet draft −236, ΔP 52 |

Full statistics are in `clusters_summary.csv`, with plots `10_cluster_scatter.png`, `11_state_share_monthly.png` and `12_state_profiles.png`. Notes:
- **No separate "start-up/shutdown" state appears**, by construction. The ramps happen with the fan off and so fall inside the detected shutdowns. The short, turbulent running periods right after restarts land mostly in S1 (Unstable).
- **The states partly encode long-term drift of the operating point.**

  | Year | Time in S2 (reduced draft) | Time in S0 (high load) |
  |---|---|---|
  | 2017 | 80 % | 7 % |
  | 2020 | 5 % | 68 % |

## 4. Contextual anomaly detection

**Method:** a separate Isolation Forest is trained per state on 13 features: the 6 sensors, inlet − outlet gas temperature, pressure drop, cone − inlet draft, the 15-minute change in inlet temperature, and the 1-hour standard deviations. A point is anomalous if either:
- **(a)** its IF score is in the top 1 % *of its own state* **and** at least one variable is more than 5 robust-σ (median/MAD) from that state's median, or
- **(b)** any level variable is more than 10 robust-σ from the state median. This path catches sustained single-sensor shifts that a multivariate IF under-scores.

**Events:** flagged points separated by ≤ 10 min are merged into events. Each event lists its most implicated variables (mean robust z) and its context.

**Result:** **746 events** covering 0.76 % of active points. 689 events came via path (a) and 57 via (b) only. The median event lasts 5 min; 86 last at least 30 min.

**Case studies:** 5 are selected deterministically, one per failure signature (highest severity, at least 7 days apart, and one forced pre-shutdown example). Plots are `14_anomaly_case_{1..5}.png`, and the auto-extracted evidence is in `outputs/anomaly_case_studies.csv`.

> **The root causes below are hypotheses inferred from sensor data only. None has been confirmed against operator logs, maintenance records or field inspection.**

| # | Event | What the data shows | Hypothesis and reasoning |
|---|---|---|---|
| 1 | **2017-03-31 04:30, 25 min**, Normal, 4 h after a restart | Material temperature drops out, then reads a **perfectly flat 650 °C** for about 40 min (−8.3 robust-σ) and jumps back to about 1,050 °C. Inlet and outlet gas temperatures and all three drafts are unchanged throughout. | **Material-temperature signal fault or substituted value (historian/DCS), not a process event.** A real flush of cold material would also cool the gas and disturb the cone draft; neither happened. A constant value is typical of a held or substituted reading. |
| 2 | **2018-03-19 05:55, 7.75 h**, Normal | **Outlet gas temperature reads 400–500 °C while inlet gas is about 920 °C** (a 400 °C drop across a cyclone that normally loses about 10–20 °C) and material is about 950 °C. Drafts are normal. It recovers gradually over the next 2 h. This is the most severe event of a cluster of **156 similar outlet-temperature events between March and June 2018** (178 such events in 2018 versus 4 in 2017). | **Outlet thermocouple degradation (e.g. damaged sheath, poor contact, coating on the thermowell) or local false-air ingress at the measuring point.** False air large enough to cool *all* the gas by 400 °C would change the outlet draft and the pressure drop, and they were unchanged. The months-long recurrence also points to an instrument that was eventually repaired or replaced, rather than a process upset. |
| 3 | **2018-05-03 21:05, 20 min**, Normal | Inlet gas temperature reads **0 °C** for several samples (−38 robust-σ), then sits **exactly flat at 800 °C**. The outlet temperature later shows the same flat 800 °C plus a single 330 °C spike. Material temperature and drafts are perfectly normal. | **Signal / transmitter or wiring fault on the temperature channels** (e.g. open-circuit thermocouple → 0, then a substituted default of 800). The process kept running, as both the drafts and the material temperature show. |
| 4 | **2019-04-13 08:30, 75 min**, Normal | **Cone draft rises steadily from about −160 to about −100 mmWC over about 6 h** (+5.2 robust-σ, toward atmospheric) while inlet and outlet drafts stay constant. At about 09:50 it **steps back by about 30 mmWC** and settles near −130. About 6 h later all drafts collapse (fan stop). | **Gradual material build-up / partial blockage in the cone or dip-tube, partly cleared at about 09:50** (e.g. by an air blaster, poking or a self-release). The pressure change is local to the cone and is not seen at the inlet or outlet, which is the classic cyclone-choking signature. The detector fired only when the deviation passed 5 σ, hours after the trend began, so a **trend rule would give earlier warning** (recommendation 3). |
| 5 | **2019-12-25 02:45, 60 min**, Reduced draft, **right before a stop** | Restart after a long stop. Drafts only reach −75 to −100 mmWC, far weaker than the state's normal −192 (inlet and outlet +6.5 robust-σ). Inlet gas is 950–1,050 °C, and material temperature keeps *falling* (no material passing the sensor yet). The drafts then collapse into a second stop (03:45–05:00), after which the restart succeeds. | **An aborted first restart attempt**: the fan/damper did not establish full suction, or the start-up sequence was interrupted, before feed was introduced. This matches the wider pattern that **30.5 % of restarts are followed by another stop within 24 h**. |

## 5. One-hour forecast of `Cyclone_Inlet_Gas_Temp`

**Setup:**
- Direct multi-horizon forecasting (h = 1…12 steps of 5 min), predicting the *change* from the current value.
- **Training data:** 2017-01-01 → 2019-12-31, minus a 1-h purge so no training target overlaps the test period. Every 2nd origin is used (137,787 origins).
- **Test data:** 2020-01-01 → 2020-08-07, using **non-overlapping hourly origins** (5,193 origins) so `forecasts.csv` reproduces the metrics exactly. Shutdowns are *included* in the test set, as in real use.
- **Features:** lags 0–12 and 18/24/36/72/288 of the target, rolling means and standard deviations, differences, current values / changes / 1-h means of the other 5 sensors, causal on/off flags (draft-off, temperature < 300 at time t), and hour of day.

| Model | RMSE (all) | MAE (all) | RMSE steady | RMSE shutdown/transition | MAE shutdown/transition |
|---|---|---|---|---|---|
| Persistence | 19.3 | 11.8 | 19.3 | 19.3 | 4.3 |
| Ridge AR-X | 16.6 | 10.3 | 15.9 | 19.3 | 7.1 |
| **HistGradientBoosting** | **15.0** | **9.2** | **15.1** | **14.5** | **3.3** |

Units are °C, averaged over all 12 horizons. Per-horizon results are in `outputs/forecast_metrics.csv` and `plots/15_forecast_error_by_horizon.png`; examples are in `plots/16_forecast_examples.png`.

**Discussion:**
- **Overall:** gradient boosting cuts RMSE by about 22 % compared with persistence (h = 1: 13.6 vs 17.4; h = 12: 16.2 vs 21.0).
- **Ridge** helps in steady operation but **does no better than persistence around stops and restarts** (RMSE 19.3 vs 19.3; its MAE there is worse). A linear model cannot represent the switch between "running" and "cooling" dynamics. Trees can, using the on/off flags.
- **Noise floor:** even 5 minutes ahead, the error is about 14–17 °C, because inlet temperature carries about ±15 °C of high-frequency variation (1-hour std ≈ 12–18 °C in normal states). That noise is the floor, not model capacity.
- **Shutdown/transition subset:** this has *low* MAE but high RMSE. Most of those origins are cold and flat, where every model is right. The errors are concentrated in a few cliff-like drops at the start of stops, which no model sees coming from temperature history alone.
- **Practical challenges:**
  - shutdowns (unpredictable cliffs)
  - regime drift across years (the training data is mostly from S2/S0 mixes that differ from 2020)
  - sensor faults inside the target itself (case 3)
  - non-stationary variance
- **Next steps:** regime-aware or quantile models, recurring retraining, using a "stop requested" signal from the DCS if one exists, and reporting prediction intervals.

## 6. Insights and recommendations

1. **Instrument health is the largest single source of "anomalies".** The most severe events (cases 1–3), and 156 outlet-temperature events in March–June 2018 alone, are sensor or signal faults rather than process upsets. → **Add instrument-health rules:** flat-line ≥ 30 min, readings of exactly 0 or round substituted values, and a **cross-check that outlet gas temperature stays within about 50 °C of inlet while running**. Alert maintenance, not operators. Log historian quality codes as well as values.
2. **Anomalies concentrate around stops and restarts.**
   - 12.9 % of events fall in the 2 h before a stop, against 1.1 % of running time (about 12× enrichment).
   - 17.7 % fall within 6 h after a restart, against 3.2 % (about 5×).
   - 57.8 % of stops had an anomaly in the preceding 2 h.

   Part of this is the onset of the stop itself (drafts weakening before they cross −60), so it is a **candidate early-warning signal that needs validation against stop causes**, not proof of prediction. → Alert when all three drafts weaken more than 3 robust-σ for the current state for ≥ 10 min.
3. **Cone choking has a visible precursor** (case 4): cone draft drifting toward atmospheric while outlet draft holds. → Add a **trend alarm on (cone − outlet) draft**, for example a rise of more than 30 mmWC over 2 h, so it fires hours before a fixed threshold would. Correlate with air-blaster or cleaning logs to confirm.
4. **Restarts are fragile and downtime is dominated by a few long stops.**
   - 30.5 % of restarts are followed by another stop within 24 h.
   - The 33 stops longer than 24 h hold about 95 % of downtime.

   → Review the start-up procedure (draft establishment before feed; case 5), and record **stop reason codes** (planned / trip / process) so availability losses can be attributed.
5. **The operating point drifts over the years.** S2 "Reduced draft" covers 80 % of 2017; S0 "High load" covers 68 % of 2020. Static thresholds and models will degrade. → **Re-baseline state models and anomaly thresholds quarterly**, and use gradient-boosted short-term forecasts (about 22 % lower RMSE than persistence) with retraining.

**Further data that would sharpen this work:**
- operator and maintenance logs (to validate events and hypotheses)
- ID-fan speed / current and damper position (to separate fan actions from process changes)
- feed rate and kiln/calciner fuel (to explain temperature excursions)
- historian quality flags

## Assumptions and limitations

- I have no ground truth for shutdowns, states or anomalies. All thresholds come from the data's own distributions and physical reasoning, and they are documented as constants at the top of the script.
- The draft sign convention is assumed to be negative = suction (consistent with the data: about −200 when running, about 0 when stopped). Units are assumed to be °C and mmWC.
- Clustering silhouette is low (0.18). The states are interpretable regimes along a continuum.
- Anomaly rates are set by design choices (top 1 % per state, robust-z gates), so absolute counts depend on those settings. The comparisons (pre-stop enrichment, signature clusters) are more meaningful than the totals.

## Plot index

| Plot | Shows |
|---|---|
| `01_missing_by_month` | Missing data by month |
| `02_distributions` | Value distributions |
| `03_correlation_matrix` | Correlations (all data / active only) |
| `04_week_slice` | Representative week |
| `05_year_slice_2018` | Representative year |
| `06_shutdowns_year_{2017,2018,2019}` | Full years with shutdowns highlighted |
| `07_downtime_monthly_and_durations` | Monthly downtime and duration histogram |
| `08_shutdown_rule_example` | Why draft rather than temperature |
| `09_cluster_model_selection` | k selection |
| `10_cluster_scatter`, `11_state_share_monthly`, `12_state_profiles` | Operating states |
| `13_anomaly_overview` | Events over time, monthly by signature |
| `14_anomaly_case_1..5` | Case-study context plots |
| `15_forecast_error_by_horizon`, `16_forecast_examples` | Forecasting |
