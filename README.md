# FatigueSet ML Pipeline

**fatigueset_parser_v1** — An end-to-end pipeline for extracting ECG/HRV features from the FatigueSet dataset and aligning them with self-reported physical fatigue labels for supervised machine learning.

## Overview

FatigueSet is a multimodal dataset containing ECG (chest strap), RR intervals, and task performance data from 12 subjects across 3 sessions each (low / medium / high physical intensity). This pipeline:

1. Loads raw ECG waveforms and RR intervals
2. Applies **task-aligned windowing** using experiment markers (baseline, activity, fatigue blocks)
3. Extracts HRV features (HR, RMSSD, SDNN, LF/HF) per window
4. Joins **block-level physical fatigue labels** from `exp_fatigue.csv` (with SSS fallback)
5. Produces a clean, labeled tabular dataset ready for model training

## Pipeline Overview

```
raw/fatigueset/{subject}/{session}/
  chest_raw_ecg.csv ──────┐
  chest_rr_interval.csv ──┤
  exp_markers.csv ────────┼──→  features/windowing.py
  exp_fatigue.csv ────────┤     (task-aligned windows, per subject/session)
  metadata.csv ───────────┤
  pre_task_survey.xlsx ───┘
                                        │
                                        ▼
                              features/ecg.py
                    (hr, rmssd, sdnn, lf_hf, data_quality)
                                        │
                                        ▼
                     label + covariate join, per window
         label:      exp_fatigue.csv block-pairing (physicalFatigueScore), SSS fallback
         covariates: sss_pretask, gvas_sleepy, intensity_level
                                        │
                                        ▼
             parser1_participant.py  →  P01.csv … P12.csv
                                        │
                                        ▼
                  parser2_combine.py  (concatenate)
                                        │
                                        ▼
                  datastore/fatigueset_final.csv
                  (677 rows, 100% label coverage)
                                        │
                                        ▼
                 eda/eda_fatigueset_edit.ipynb
```

**What happens at each stage:**

1. **Raw inputs** — per-session sensor files (ECG waveform, RR intervals) and per-session experiment logs (task markers, fatigue check-ins), alongside root-level baseline surveys (SSS/GVAS) and the subject→intensity mapping.
2. **Windowing** — `features/windowing.py` slices each session into fixed-length windows (30s), aligned to task-block boundaries from `exp_markers.csv`. Windows are defined purely from marker geometry — no sensor file's health can zero out a session's window count.
3. **Feature extraction** — `features/ecg.py` extracts HR, RMSSD, SDNN from RR intervals inside each window. LF/HF is computed on separate 2-minute (120s) windows and inherited by every 30s window that falls within it (~4 rows share one identical lf_hf value). If a window's ECG waveform slice is empty/corrupted, `data_quality` is set to `unknown` for that window only.
4. **Label + covariate join** — each window's task block is matched to its corresponding `exp_fatigue.csv` measurement (block-level pairing); sessions or blocks with no fatigue measurement fall back to the pre-task SSS value. Session-level covariates (`sss_pretask`, `gvas_sleepy`, `intensity_level`) are broadcast onto every window belonging to that session.
5. **Per-subject assembly** — `parser1_participant.py` runs this full chain once per subject (steps 1–4), writing one labeled CSV per subject.
6. **Combine** — `parser2_combine.py` concatenates all 12 subject CSVs into a single dataset.
7. **EDA** — `eda_fatigueset_edit.ipynb` consumes the final CSV for label distribution, missing-data checks, and feature range sanity checks.

## Formulas & Label Construction

### ECG/HRV features (per window)

Every window's `hr`/`rmssd`/`sdnn` come from the R-R intervals (ms) falling inside that window's time range. LF/HF uses a separate 2-minute window (see below).

**HR — heart rate**
```
HR = 60000 / mean(RR)
```
e.g. a window with mean RR = 1081.07 ms → `HR = 60000 / 1081.07 = 55.5 bpm`.

**SDNN — standard deviation of RR intervals**
```
SDNN = sqrt( (1/n) * Σ (RR_i - mean(RR))^2 )
```
Population standard deviation of all RR durations in the window. Captures overall spread across the whole window.

**RMSSD — root mean square of successive differences**
```
RMSSD = sqrt( (1/(n-1)) * Σ (RR_(i+1) - RR_i)^2 )
```
Takes the diff between each beat and the *next* beat, squares it, averages, square-roots. Reacts to fast beat-to-beat jitter; SDNN and RMSSD are not redundant — they pick up different timescales of variability.

**LF/HF — frequency-domain HRV ratio (2-minute windows)**
```
1. Define 120-second windows (lf_hf_window_seconds) from the session timeline
2. Extract RR intervals within each 2-minute window
3. If valid RR duration < 90 seconds (lf_hf_min_confidence_seconds), set lf_hf_low_confidence = True
4. Run a Lomb-Scargle periodogram (handles unevenly-spaced samples)
5. Sum power in LF band (0.04–0.15 Hz) and HF band (0.15–0.4 Hz)
6. LF/HF = LF_power / HF_power
7. Every 30s window inherits the lf_hf value from whichever 2-minute window it falls in
```
With only ~25–30 beats in a 30s window, this ratio is noisy by nature — treat it as a rough covariate more than a precise measurement. The 2-minute window provides ~100 beats for a more stable estimate.

**`data_quality` — artefact flag**
```
data_quality = "unknown" if ecg_waveform is unavailable/corrupted (e.g. P01 S1)
data_quality = "valid"   otherwise
```
P01 Session 1 has corrupted ECG timestamps (all rows identical `1.62937E+12`), so its 27 windows are marked `unknown`. This is a per-window flag — other sessions and subjects are unaffected.

### Label construction — `exp_fatigue.csv`, block-level pairing

1. **Task blocks** are defined by `exp_markers.csv` boundaries: each block has a start and end time (seconds elapsed since session start).
2. **Assign each window to a block**:
   ```
   block_id(window) = the block b such that
       block_b.start_sec <= window.window_start_sec < block_b.end_sec
   ```
3. **Assign each `exp_fatigue.csv` measurement to a block** the same way, using `fatigueSurveySubmissionTime` — a measurement taken during or right after a block is treated as that block's fatigue rating (block-level pairing, not a per-window nearest-timestamp match).
4. **Every window inherits its block's fatigue score**:
   ```
   label(window) = physicalFatigueScore of the exp_fatigue.csv row
                   paired with block_id(window)
   ```
   This is what produces 100% label coverage (677/677 rows) — instead of only the ~3 windows per session nearest each raw measurement getting a value, the whole block those measurements represent gets labeled.
5. **Fallback**: if a session/block has no `exp_fatigue.csv` measurement at all, `label` = the session's Stanford Sleepiness Scale value instead, and `label_type` switches from `"fatigue_rating"` to `"sss_pretask"` for those rows.

### Stanford Sleepiness Scale — text-to-number mapping

`pre_task_survey.xlsx`'s sleepiness field is stored as text, not a number, so it's converted via a fixed lookup:
```
"Feeling active and vital; alert; wide awake."                          -> 1
"Functioning at a high level, but not at peak; able to concentrate."    -> 2
"Relaxed; awake; not at full alertness; responsive."                     -> 3
"A little foggy; not at peak; let down."                                 -> 4
```
(Levels 5–7 — foggy/sleepy/fighting sleep — extend this map if they appear once the full dataset is loaded; not observed in the initial sample.)

### GVAS sleepiness covariate

`"How sleepy do you feel?"` is already numeric (0–6) in `pre_task_survey.xlsx` — no conversion, just carried through as `gvas_sleepy`. It's a separate scale from the SSS above, despite the similar wording — don't treat them as the same measurement.

### Intensity level marker

`metadata.csv` stores, per participant, which session number was low/medium/high intensity:
```
participant_id, low_session, medium_session, high_session
```
This gets inverted per subject into a lookup:
```
{low_session: "low", medium_session: "medium", high_session: "high"}
```
so each `session_id` maps directly to an `intensity_level` string, broadcast to every window in that session.

## Project Structure

```
.
├── configs/
│   └── datasets/
│       └── fatigueset.yaml          # Dataset configuration
├── parsers/
│   └── fatigueset/
│       ├── parser1_participant.py   # Per-subject processing
│       └── parser2_combine.py       # Combine into final CSV
├── features/
│   ├── ecg.py                       # ECG/HRV feature computation
│   └── windowing.py                 # Task-aligned windowing
├── eda/
│   └── eda_fatigueset_edit.ipynb    # Exploratory data analysis notebook
├── raw/
│   ├── fatigueset/                  # Raw sensor data (not tracked in git)
│   └── fatigueset_participants/     # Per-subject pipeline output CSVs
├── datastore/
│   └── fatigueset_final.csv         # Final combined dataset
└── master_fatigueset.md             # Internal specification (not published)
```

## Raw Data Layout

```
raw/fatigueset/
├── metadata.csv                     # Subject → session intensity mapping
├── pre_task_survey.xlsx             # Baseline SSS + GVAS (fallback labels)
├── 01/
│   ├── 01/
│   │   ├── chest_raw_ecg.csv        # ECG waveform (timestamp_ms, ecg_waveform)
│   │   ├── chest_rr_interval.csv    # RR intervals (timestamp_ms, duration)
│   │   ├── exp_fatigue.csv          # Primary fatigue labels
│   │   ├── exp_markers.csv          # Task block timestamps
│   │   ├── exp_crt.csv              # Choice reaction time (deferred)
│   │   ├── exp_nback.csv            # N-back (deferred)
│   │   └── exp_task_switch.csv      # Task switching (deferred)
│   ├── 02/                          # Session 2
│   └── 03/                          # Session 3
├── 02/ ... 12/                      # Subjects 2–12
```

## Pipeline Steps

### Step 1: Per-Subject Processing

```bash
python parsers/fatigueset/parser1_participant.py --subject 1
python parsers/fatigueset/parser1_participant.py --subject 2
# ... for subjects 1–12
```

Optional arguments:
- `--config` — Path to YAML config (default: `configs/datasets/fatigueset.yaml`)
- `--output-dir` — Output directory (default: `raw/fatigueset_participants`)
- `--log-level` — DEBUG, INFO, WARNING, ERROR

Each run produces a per-subject CSV (`P01.csv`, `P02.csv`, ...) with ~45–72 rows.

### Step 2: Combine into Final Dataset

```bash
python parsers/fatigueset/parser2_combine.py
```

Output: `datastore/fatigueset_final.csv` (677 rows, 100% label coverage)


## Configuration (`configs/datasets/fatigueset.yaml`)

| Key | Description |
|-----|-------------|
| `raw_root` | Path to raw data |
| `window_seconds` | Window length (30s) |
| `overlap_seconds` | Window overlap (0 = non-overlapping) |
| `use_task_markers` | Enable task-aligned windowing |
| `ecg_sample_rate_hz` | ECG sampling rate (250 Hz) |
| `invalid_ecg_value` | ECG invalid sample marker (4095) |
| `label_type` | Label source name |
| `fatigue_timestamp_tolerance_seconds` | Timestamp matching tolerance |
| `lf_hf_window_seconds` | LF/HF computation window (120s) |
| `lf_hf_min_confidence_seconds` | Minimum RR data for reliable LF/HF (90s) |

## Features (per-window)

| Feature | Description |
|---------|-------------|
| `hr` | Mean heart rate (bpm) |
| `rmssd` | Root mean square of successive RR differences (ms) |
| `sdnn` | Standard deviation of NN intervals (ms) |
| `lf_hf` | Low frequency / high frequency power ratio (2-min window, inherited) |
| `lf_hf_window_sec` | Actual duration of the 2-min LF/HF window used |
| `lf_hf_low_confidence` | True if RR data < 90s in the LF/HF window |
| `data_quality` | `valid` or `unknown` (corrupted ECG) |

## Labeling Strategy

- **Primary source:** `physicalFatigueScore` from `exp_fatigue.csv` → `label` (0–100 scale)
- **Matching method:** Block-level pairing — task blocks come from `exp_markers.csv`; each `exp_fatigue.csv` measurement is assigned to the block it falls in, and every window in that block inherits the score (not just the single nearest window).
- **Fallback:** For any block with no paired `exp_fatigue.csv` measurement, `label` falls back to that session's Stanford Sleepiness Scale (SSS) value from `pre_task_survey.xlsx`, and `label_type` is set to `"sss_pretask"` for just those rows (vs `"fatigue_rating"` where a real measurement applied).

## Output Columns (final CSV)

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | str | P01–P12 |
| `session_id` | int | 1, 2, or 3 |
| `window_start_sec` | float | Seconds from session start |
| `hr` | float | Mean heart rate (bpm) |
| `rmssd` | float | HRV metric (ms) |
| `sdnn` | float | HRV metric (ms) |
| `lf_hf` | float | LF/HF ratio (2-min window, inherited) |
| `lf_hf_window_sec` | float | Actual duration of LF/HF computation window |
| `lf_hf_low_confidence` | bool | True if RR data < 90s in LF/HF window |
| `data_quality` | str | `valid` or `unknown` |
| `label` | float | Physical fatigue score (0–100) or SSS fallback |
| `label_type` | str | `fatigue_rating` or `sss_pretask` |
| `sss_pretask` | float | Baseline sleepiness (1–4) |
| `gvas_sleepy` | float | GVAS sleepiness score (0–6) |
| `intensity_level` | str | `low` / `medium` / `high` |
| `modality` | str | `ecg` |
| `dataset` | str | `fatigueset` |

## Results

- **677 total windows** across 12 subjects × 3 sessions
- **100% label coverage** (0 NaN, thanks to block-level pairing + SSS fallback)
- **`label` range:** 0.00–77.22 (mean ≈ 34)
- **`label_type`:** 677/677 `fatigue_rating` (all blocks paired, no SSS fallback needed)
- **`data_quality`:** 650 `valid`, 27 `unknown` (P01 S1 — corrupted ECG timestamps)
- **`lf_hf_low_confidence`:** 157/677 windows (23%) with < 90s RR data in their 2-min LF/HF window
- **Balanced intensity levels:** medium (228), high (227), low (222)

## Dependencies

- Python ≥ 3.10
- pandas, numpy, scipy
- matplotlib, seaborn
- pyyaml
- openpyxl
- jupyter (for EDA notebook)

```bash
pip install pandas numpy scipy matplotlib seaborn pyyaml openpyxl jupyter
```

## License

This project uses the **FatigueSet** dataset. See the original dataset license for terms of use.
