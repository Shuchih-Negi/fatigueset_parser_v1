# FatigueSet ML Pipeline

**fatigueset_parser_v1** — An end-to-end pipeline for extracting ECG/HRV features from the FatigueSet dataset and aligning them with self-reported physical fatigue labels for supervised machine learning.

## Overview

FatigueSet is a multimodal dataset containing ECG (chest strap), RR intervals, and task performance data from 12 subjects across 3 sessions each (low / medium / high physical intensity). This pipeline:

1. Loads raw ECG waveforms and RR intervals
2. Applies **task-aligned windowing** using experiment markers (baseline, activity, fatigue blocks)
3. Extracts HRV features (HR, RMSSD, SDNN, LF/HF) per window
4. Joins **block-level physical and mental fatigue labels** from `exp_fatigue.csv` (with SSS fallback)
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
         label:      exp_fatigue.csv block-pairing (label_physical + label_mental), SSS fallback
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
                  (749 rows, 100% label coverage)
                                        │
                                        ▼
                 eda/eda_fatigueset_final.ipynb
```

**What happens at each stage:**

1. **Raw inputs** — per-session sensor files (ECG waveform, RR intervals) and per-session experiment logs (task markers, fatigue check-ins), alongside root-level baseline surveys (SSS/GVAS) and the subject→intensity mapping.
2. **Windowing** — `features/windowing.py` slices each session into fixed-length windows, aligned to task-block boundaries from `exp_markers.csv` rather than raw wall-clock time, so no window straddles a task transition.
3. **Feature extraction** — `features/ecg.py` turns the RR intervals (and raw ECG waveform, for artefact detection) inside each window into `hr`, `rmssd`, `sdnn`, `lf_hf`, plus a `data_quality` flag.
4. **Label + covariate join** — each window's task block is matched to its corresponding `exp_fatigue.csv` measurement (block-level pairing); sessions or blocks with no fatigue measurement fall back to the pre-task SSS value. Session-level covariates (`sss_pretask`, `gvas_sleepy`, `intensity_level`) are broadcast onto every window belonging to that session.
5. **Per-subject assembly** — `parser1_participant.py` runs this full chain once per subject (steps 1–4), writing one labeled CSV per subject.
6. **Combine** — `parser2_combine.py` concatenates all 12 subject CSVs into a single dataset.
7. **EDA** — `eda_fatigueset_final.ipynb` consumes the final CSV for label distribution, missing-data checks, and feature range sanity checks.

## Formulas & Label Construction

### ECG/HRV features (per window)

Every window's `hr`/`rmssd`/`sdnn`/`lf_hf` come from the R-R intervals (ms) falling inside that window's time range.

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

**LF/HF — frequency-domain HRV ratio**
RR intervals aren't evenly spaced in time, so this isn't a plain FFT:
```
1. Treat the window as (beat_timestamp, RR_duration) pairs
2. Run a Lomb-Scargle periodogram (handles unevenly-spaced samples)
3. Sum power in LF band (0.04–0.15 Hz) and HF band (0.15–0.4 Hz)
4. LF/HF = LF_power / HF_power
```
With only ~25–30 beats in a 30s window, this ratio is noisy by nature — treat it as a rough covariate more than a precise measurement.

**`data_quality` — artefact flag**
```
artefact_fraction = (count of invalid samples, ecg_waveform == 4095) / (total samples in window)
data_quality = "artefact" if artefact_fraction > threshold else "valid"
```
Not a fatigue feature — a gate on whether to trust the above numbers for that window.

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
   This is what produces 100% label coverage (749/749 rows) — instead of only the ~3 windows per session nearest each raw measurement getting a value, the whole block those measurements represent gets labeled.
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
│   └── windowing.py                 # Sliding + task-aligned windowing
├── eda/
│   └── eda_fatigueset_final.ipynb   # Exploratory data analysis notebook
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

Each run produces a per-subject CSV (`P01.csv`, `P02.csv`, ...) with ~50–70 rows.

### Step 2: Combine into Final Dataset

```bash
python parsers/fatigueset/parser2_combine.py
```

Output: `datastore/fatigueset_final.csv` (749 rows, 100% label coverage)


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

## Features (per-window)

| Feature | Description |
|---------|-------------|
| `hr` | Mean heart rate (bpm) |
| `rmssd` | Root mean square of successive RR differences (ms) |
| `sdnn` | Standard deviation of NN intervals (ms) |
| `lf_hf` | Low frequency / high frequency power ratio |
| `data_quality` | Artifact flag (`valid` or `invalid`) |

## Labeling Strategy

- **Primary sources (co-primary, both kept):**
  - `physicalFatigueScore` from `exp_fatigue.csv` → `label_physical` (0–100 scale)
  - `mentalFatigueScore` from `exp_fatigue.csv` → `label_mental` (0–100 scale)
  - Neither is dropped in favor of the other — `physicalFatigueScore` tracks the low/medium/high intensity manipulation closely, while `mentalFatigueScore` is the target the original project roadmap specified for this dataset.
- **Matching method:** Block-level pairing — task blocks come from `exp_markers.csv`; each `exp_fatigue.csv` measurement is assigned to the block it falls in, and every window in that block inherits both scores (not just the single nearest window).
- **Fallback:** For any block with no paired `exp_fatigue.csv` measurement, both `label_physical` and `label_mental` fall back to that session's Stanford Sleepiness Scale (SSS) value from `pre_task_survey.xlsx`, and `label_type` is set to `"sss_pretask"` for just those rows (vs `"fatigue_rating"` where a real measurement applied).

## Output Columns (final CSV)

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | str | P01–P12 |
| `session_id` | int | 1, 2, or 3 |
| `window_start_sec` | float | Seconds from session start |
| `hr` | float | Mean heart rate |
| `rmssd` | float | HRV metric |
| `sdnn` | float | HRV metric |
| `lf_hf` | float | HRV metric |
| `data_quality` | str | `valid` / `invalid` |
| `label_physical` | float | Physical fatigue score (0–100) or SSS fallback |
| `label_mental` | float | Mental fatigue score (0–100) or SSS fallback |
| `label_type` | str | `fatigue_rating` or `sss_pretask` |
| `sss_pretask` | float | Baseline sleepiness (1–7) |
| `gvas_sleepy` | float | GVAS sleepiness score |
| `intensity_level` | str | `low` / `medium` / `high` |
| `modality` | str | `ecg` |
| `dataset` | str | `fatigueset` |

## Results

- **749 total windows** across 12 subjects × 3 sessions
- **100% label coverage** (0 NaN, thanks to block-level pairing + SSS fallback)
- **`label_physical` range:** 0.00–77.22 (mean ≈ 34)
- **`label_mental` range:** not yet summarized here — re-run the EDA notebook's distribution cell for `label_mental` before reporting, since it can diverge meaningfully from the physical score (see `eda_fatigueset_final.ipynb`)
- **Balanced intensity levels:** high (263), medium (261), low (225)

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
