# FatigueSet ML Pipeline

**fatigueset_parser_v1** — An end-to-end pipeline for extracting ECG/HRV features from the FatigueSet dataset and aligning them with self-reported physical fatigue labels for supervised machine learning.

## Overview

FatigueSet is a multimodal dataset containing ECG (chest strap), RR intervals, and task performance data from 12 subjects across 3 sessions each (low / medium / high physical intensity). This pipeline:

1. Loads raw ECG waveforms and RR intervals
2. Applies **task-aligned windowing** using experiment markers (baseline, activity, fatigue blocks)
3. Extracts HRV features (HR, RMSSD, SDNN, LF/HF) per window
4. Joins **block-level fatigue labels** from `exp_fatigue.csv` (with SSS fallback)
5. Produces a clean, labeled tabular dataset ready for model training

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

### Step 3: Exploratory Data Analysis

Open `eda/eda_fatigueset_final.ipynb` in Jupyter Lab for visual exploration, correlation analysis, and distribution checks.

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

- **Primary source:** `exp_fatigue.csv` — `physicalFatigueScore` (0–100 scale)
- **Matching method:** Block-level pairing — each measurement applies to the preceding task block
- **Fallback:** When `exp_fatigue.csv` is missing, Stanford Sleepiness Scale (SSS) from `pre_task_survey.xlsx` is broadcast per-session

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
| `label` | float | Fatigue score or NaN |
| `label_type` | str | `fatigue_rating` or `sss_pretask` |
| `sss_pretask` | float | Baseline sleepiness (1–7) |
| `gvas_sleepy` | float | GVAS sleepiness score |
| `intensity_level` | str | `low` / `medium` / `high` |
| `modality` | str | `ecg` |
| `dataset` | str | `fatigueset` |

## Results

- **749 total windows** across 12 subjects × 3 sessions
- **100% label coverage** (0 NaN labels)
- **Label range:** 0.00–77.22 (mean ≈ 34)
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
