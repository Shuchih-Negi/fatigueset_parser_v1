"""
Parser 1: Process individual FatigueSet participant.

Loads ECG/RR interval data, applies windowing and feature extraction,
joins labels from exp_fatigue.csv and metadata, outputs one CSV per subject.

=============================================================================
Section 4 resolution — exp_ file inspection findings (master_fatigueset.md):
=============================================================================

exp_fatigue.csv (PRIMARY LABEL SOURCE):
  Columns: measurementNumber, physicalFatigueAnswerTime, mentalFatigueAnswerTime,
           fatigueSurveySubmissionTime, physicalFatigueScore, mentalFatigueScore
  Timestamps in seconds from session start. ~3 measurements per session.
  physicalFatigueScore: continuous rating (0-100 scale).
  Verdict: USABLE as per-window label source — joined by nearest-timestamp match.

exp_markers.csv (TASK BLOCK BOUNDARIES — used for window alignment):
  Columns: utcTime (ms), eventMarker (string)
  Markers include: start_baseline, end_baseline, start_activity, end_activity,
                   start_fatigue, end_fatigue, start_crt, start_nback, etc.
  Verdict: CONFIRMED as task block boundaries — windows are sliced within
           block boundaries, skipping inter-block gaps (when use_task_markers=true).

exp_crt.csv (Choice Reaction Time — deferred):
  Columns: trialStartTime, stimulusStartTime, stimulusCoords, correctResponse,
           participantResponse, isCorrectResponse, responseTime, ...
  Verdict: Per-trial cognitive data. Deferred to future features/behavior.py
           for reaction-time drift / error-rate drift analysis.

exp_nback.csv (N-back — deferred):
  Columns: trialStartTime, trialEndTime, stimulus, correctResponse,
           participantResponse, isCorrectResponse, responseTime, ...
  Verdict: Per-trial working memory data. Deferred to future features/behavior.py.

exp_task_switch.csv (Task Switching — deferred):
  Columns: trialStartTime, trialNumber, repNumber, stimulusStartTime,
           stimulusQuadrant, stimulus, correctResponse, ...
  Verdict: Per-trial executive function data. Deferred to future features/behavior.py.

=============================================================================
Label fallback behavior:
  Primary: exp_fatigue.csv (per-window, joined by timestamp).
  Fallback: If exp_fatigue.csv is missing/empty for a session, the Stanford
  Sleepiness Scale (SSS) from pre_task_survey.xlsx is used instead, broadcast
  uniformly across all windows in that session (per_session scope). The
  label_type changes to "sss_pretask" for fallback rows.
=============================================================================

Usage:
    python parser1_participant.py --subject 1
    python parser1_participant.py --subject 1 --config configs/datasets/fatigueset.yaml
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

import pandas as pd
import numpy as np
import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features import windowing, ecg
from features.windowing import extract_task_blocks

logger = logging.getLogger(__name__)


def build_intensity_lookup(
    subject_id: int, metadata_df: pd.DataFrame
) -> Dict[int, str]:
    """
    Build session -> intensity_level mapping for one subject.

    Args:
        subject_id: subject number (1-12)
        metadata_df: DataFrame with columns [participant_id, low_session, medium_session, high_session]

    Returns:
        dict mapping session_id (1-3) to intensity_level ("low", "medium", "high")
    """
    row = metadata_df[metadata_df["participant_id"] == subject_id]
    if len(row) == 0:
        logger.warning(f"No metadata row found for subject {subject_id}")
        return {1: np.nan, 2: np.nan, 3: np.nan}

    row = row.iloc[0]
    intensity_lookup = {}
    intensity_lookup[int(row["low_session"])] = "low"
    intensity_lookup[int(row["medium_session"])] = "medium"
    intensity_lookup[int(row["high_session"])] = "high"

    return intensity_lookup


def load_exp_fatigue_lookup(
    subject_id: int, session_id: int, raw_root: Path
) -> pd.DataFrame:
    """
    Load fatigue survey data for one subject/session.

    Args:
        subject_id: subject number (1-12)
        session_id: session number (1-3)
        raw_root: root path to raw/fatigueset/

    Returns:
        DataFrame with columns [measurementNumber, physicalFatigueAnswerTime, ..., 
                                fatigueSurveySubmissionTime, physicalFatigueScore, mentalFatigueScore]
    """
    fatigue_path = (
        raw_root / f"{subject_id:02d}" / f"{session_id:02d}" / "exp_fatigue.csv"
    )
    if not fatigue_path.exists():
        logger.warning(f"No exp_fatigue.csv found at {fatigue_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(fatigue_path)
        return df
    except Exception as e:
        logger.warning(f"Failed to load {fatigue_path}: {e}")
        return pd.DataFrame()


def load_task_markers(
    subject_id: int, session_id: int, raw_root: Path
) -> pd.DataFrame:
    """
    Load task marker data for one subject/session.

    exp_markers.csv contains task block start/stop timestamps
    (e.g., start_baseline, end_baseline, start_activity, end_activity)
    used for task-aligned windowing.

    Args:
        subject_id: subject number (1-12)
        session_id: session number (1-3)
        raw_root: root path to raw/fatigueset/

    Returns:
        DataFrame with columns [utcTime, eventMarker] or empty if not found.
        utcTime is in milliseconds (matches ECG timestamp units).
    """
    markers_path = (
        raw_root / f"{subject_id:02d}" / f"{session_id:02d}" / "exp_markers.csv"
    )
    if not markers_path.exists():
        logger.debug(f"No exp_markers.csv found at {markers_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(markers_path)
        return df
    except Exception as e:
        logger.warning(f"Failed to load {markers_path}: {e}")
        return pd.DataFrame()


def get_fatigue_for_window(
    window_start_sec_from_session_start: float,
    window_end_sec_from_session_start: float,
    fatigue_df: pd.DataFrame,
    timestamp_field: str = "fatigueSurveySubmissionTime",
    value_field: str = "physicalFatigueScore",
    tolerance_seconds: float = 5.0,
) -> float:
    """
    Get fatigue score for a window by nearest timestamp.

    Joins fatigue scores by finding the measurement whose timestamp is
    closest to the window start (within tolerance).

    Args:
        window_start_sec_from_session_start: window start in seconds from session start
        window_end_sec_from_session_start: window end in seconds from session start
        fatigue_df: DataFrame with at least [timestamp_field, value_field]
                   (fatigue timestamps are relative to session start in seconds)
        timestamp_field: column name for fatigue measurement timestamps (seconds)
        value_field: column name for fatigue score
        tolerance_seconds: max distance from window_start to match (seconds)

    Returns:
        fatigue score (numeric) or NaN if no match found
    """
    if fatigue_df.empty or timestamp_field not in fatigue_df.columns:
        return np.nan

    # Find measurements within tolerance of window_start
    timestamps_sec = fatigue_df[timestamp_field].values
    distances = np.abs(timestamps_sec - window_start_sec_from_session_start)
    closest_idx = np.argmin(distances)

    if distances[closest_idx] <= tolerance_seconds:
        return fatigue_df[value_field].iloc[closest_idx]
    else:
        return np.nan


def load_survey_data(
    subject_id: int,
    session_id: int,
    survey_path: Path,
    sss_field: str,
    gvas_field: str,
    sss_map: Dict[str, int],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Load pre-task survey data (SSS label and GVAS covariate).

    Args:
        subject_id: subject number
        session_id: session number
        survey_path: path to pre_task_survey.xlsx
        sss_field: column name for Stanford Sleepiness Scale
        gvas_field: column name for GVAS "sleepy" question
        sss_map: dict mapping SSS text to numeric value (1-7)

    Returns:
        (sss_label, gvas_score) or (NaN, NaN) if no match
    """
    if not survey_path.exists():
        logger.warning(f"Survey file not found: {survey_path}")
        return np.nan, np.nan

    try:
        df = pd.read_excel(survey_path)
        row = df[(df["ID"] == subject_id) & (df["Session"] == session_id)]

        if len(row) == 0:
            logger.warning(
                f"No survey row found for subject {subject_id}, session {session_id}"
            )
            return np.nan, np.nan

        row = row.iloc[0]

        # Parse SSS (text -> number)
        sss_text = row.get(sss_field, np.nan)
        sss_label = sss_map.get(sss_text, np.nan)
        if pd.isna(sss_label):
            logger.warning(
                f"SSS text not in map for subject {subject_id}, session {session_id}: {sss_text}"
            )

        # Parse GVAS (already numeric)
        gvas_score = row.get(gvas_field, np.nan)

        return sss_label, gvas_score

    except Exception as e:
        logger.warning(f"Failed to load survey data: {e}")
        return np.nan, np.nan


def process_participant(
    subject_id: int,
    config: Dict,
    metadata_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Process one participant (all 3 sessions).

    Loads ECG/RR data, applies windowing and feature extraction,
    joins fatigue labels and metadata, writes output CSV.

    Args:
        subject_id: subject number (1-12)
        config: configuration dict loaded from fatigueset.yaml
        metadata_df: metadata DataFrame
        output_dir: output directory for per-participant CSVs
    """
    raw_root = Path(config["raw_root"])
    window_seconds = config["window_seconds"]
    overlap_seconds = config["overlap_seconds"]
    invalid_ecg_value = config["invalid_ecg_value"]
    fatigue_tolerance = config["fatigue_timestamp_tolerance_seconds"]
    use_task_markers = config.get("use_task_markers", False)
    ecg_sample_rate_hz = config.get("ecg_sample_rate_hz", 250.0)

    # Build intensity lookup for this subject
    intensity_lookup = build_intensity_lookup(subject_id, metadata_df)

    # Load survey (pre-task baseline)
    survey_path = raw_root / config["survey_file"]
    sss_map = config.get("sss_text_map", {})
    sss_field = config.get("survey_sss_field", "Please indicate your current level of sleepiness.")
    gvas_field = config.get("survey_gvas_field", "How sleepy do you feel?")

    # Accumulate all rows for this subject
    all_rows = []

    for session_id in [1, 2, 3]:
        logger.info(f"Processing subject {subject_id}, session {session_id}")

        # Load ECG and RR data
        ecg_path = (
            raw_root / f"{subject_id:02d}" / f"{session_id:02d}" / config["ecg_waveform_file"]
        )
        rr_path = (
            raw_root / f"{subject_id:02d}" / f"{session_id:02d}" / config["rr_interval_file"]
        )

        if not ecg_path.exists() or not rr_path.exists():
            logger.error(f"Missing ECG or RR file for subject {subject_id}, session {session_id}")
            continue

        try:
            ecg_df = pd.read_csv(ecg_path)
            rr_df = pd.read_csv(rr_path)
        except Exception as e:
            logger.error(f"Failed to load ECG/RR data: {e}")
            continue

        # Load fatigue labels (primary source: exp_fatigue.csv)
        fatigue_df = load_exp_fatigue_lookup(subject_id, session_id, raw_root)
        use_exp_fatigue = not fatigue_df.empty

        # Load task markers if enabled (normalized inside try block after session_start_ms)
        markers_df_raw = load_task_markers(subject_id, session_id, raw_root) if use_task_markers else pd.DataFrame()

        # Load survey data (baseline SSS + GVAS)
        sss_pretask, gvas_pretask = load_survey_data(
            subject_id, session_id, survey_path, sss_field, gvas_field, sss_map
        )

        # Get intensity level for this session
        intensity_level = intensity_lookup.get(session_id, np.nan)

        # Prepare signal dict for windowing
        # Timestamps: ECG/RR are in milliseconds (Unix epoch)
        # Fatigue timestamps are in seconds (relative to session start)
        # Strategy: normalize all to seconds from ECG start
        try:
            ecg_timestamps_ms = ecg_df["timestamp"].values
            ecg_waveform = ecg_df["ecg_waveform"].values
            rr_timestamps_ms = rr_df["timestamp"].values
            rr_durations = rr_df["duration"].values

            # Use ECG start as session start (time 0)
            session_start_ms = ecg_timestamps_ms.min()
            
            # Convert ECG timestamps to seconds from session start
            ecg_timestamps_sec = (ecg_timestamps_ms - session_start_ms) / 1000.0
            
            # Convert RR timestamps to seconds from session start
            rr_timestamps_sec = (rr_timestamps_ms - session_start_ms) / 1000.0

            # Normalize marker timestamps: utcTime (ms) -> seconds from session start
            markers_df = markers_df_raw.copy() if not markers_df_raw.empty else pd.DataFrame()
            if not markers_df.empty:
                markers_df["utcTime"] = (markers_df["utcTime"].values.astype(float) - session_start_ms) / 1000.0
                logger.debug(f"Loaded {len(markers_df)} task markers for subject {subject_id}, session {session_id}")

            signals_dict = {
                "ecg_waveform": ecg_waveform,
            }

            # Pre-compute block-level fatigue mapping (for task-aligned windowing)
            # Each fatigue measurement applies to the task block that precedes it
            block_fatigue_map = {}
            if use_exp_fatigue and use_task_markers and not markers_df.empty:
                blocks = extract_task_blocks(markers_df)
                # Sort fatigue measurements by timestamp
                fatigue_sorted = fatigue_df.sort_values(
                    config.get("exp_fatigue_timestamp_field", "fatigueSurveySubmissionTime")
                ).reset_index(drop=True)
                # Pair blocks sequentially with fatigue measurements
                for i, (b_start, b_end) in enumerate(blocks):
                    if i < len(fatigue_sorted):
                        block_fatigue_map[i] = fatigue_sorted.iloc[i][
                            config.get("exp_fatigue_physical_field", "physicalFatigueScore")
                        ]
                    else:
                        block_fatigue_map[i] = np.nan
                logger.debug(
                    f"Block-fatigue mapping: { {k: round(v, 1) if not pd.isna(v) else None for k, v in block_fatigue_map.items()} }"
                )

            # Generate windows (timestamps are already in seconds from session start)
            for window_start_sec, window_end_sec, block_idx, window_data in windowing.window_generator(
                ecg_timestamps_sec,
                signals_dict,
                window_seconds,
                overlap_seconds,
                task_markers=markers_df if use_task_markers else None,
            ):
                # Get RR intervals for this window
                rr_mask = (rr_timestamps_sec >= window_start_sec) & (rr_timestamps_sec < window_end_sec)
                rr_in_window = rr_durations[rr_mask]

                if len(rr_in_window) == 0:
                    logger.debug(f"No RR intervals in window [{window_start_sec}, {window_end_sec}]")
                    continue

                # Compute ECG/HRV features
                ecg_waveform_window = window_data.get("ecg_waveform", np.array([]))
                features = ecg.compute_features(
                    rr_in_window,
                    ecg_waveform=ecg_waveform_window,
                    invalid_value=invalid_ecg_value,
                    sample_rate_hz=ecg_sample_rate_hz,
                )

                # Determine label: prefer exp_fatigue, fall back to SSS
                if use_exp_fatigue:
                    if use_task_markers and block_idx >= 0 and block_idx in block_fatigue_map:
                        # Task-aligned: use block-level fatigue mapping
                        window_label = block_fatigue_map[block_idx]
                    else:
                        # Per-window: match by nearest timestamp
                        window_label = get_fatigue_for_window(
                            window_start_sec,
                            window_end_sec,
                            fatigue_df,
                            timestamp_field=config.get("exp_fatigue_timestamp_field", "fatigueSurveySubmissionTime"),
                            value_field=config.get("exp_fatigue_physical_field", "physicalFatigueScore"),
                            tolerance_seconds=fatigue_tolerance,
                        )
                    window_label_type = config.get("label_type", "fatigue_rating")
                else:
                    # Fallback path: per-session SSS broadcast
                    if not pd.isna(sss_pretask):
                        logger.warning(
                            f"exp_fatigue.csv missing for subject {subject_id}, "
                            f"session {session_id} — falling back to SSS pretask label"
                        )
                    window_label = sss_pretask
                    window_label_type = "sss_pretask"

                # Assemble row
                row = {
                    "subject_id": f"P{subject_id:02d}",
                    "session_id": session_id,
                    "window_start_sec": window_start_sec,  # seconds from session start
                    "hr": features["hr"],
                    "rmssd": features["rmssd"],
                    "sdnn": features["sdnn"],
                    "lf_hf": features["lf_hf"],
                    "data_quality": features["data_quality"],
                    "label": window_label,
                    "label_type": window_label_type,
                    "sss_pretask": sss_pretask,
                    "gvas_sleepy": gvas_pretask,
                    "intensity_level": intensity_level,
                    "modality": "ecg",
                    "dataset": "fatigueset",
                }
                all_rows.append(row)

        except Exception as e:
            logger.error(f"Error processing session {session_id}: {e}")
            continue

    if len(all_rows) == 0:
        logger.error(f"No data for subject {subject_id}")
        return

    # Write output CSV
    output_df = pd.DataFrame(all_rows)
    output_path = output_dir / f"P{subject_id:02d}.csv"
    output_df.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(output_df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process one FatigueSet participant and output per-subject CSV."
    )
    parser.add_argument("--subject", type=int, required=True, help="Subject ID (1-12)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets/fatigueset.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="raw/fatigueset_participants",
        help="Output directory for per-subject CSVs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load metadata
    metadata_path = Path(config["raw_root"]) / config["metadata_file"]
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        sys.exit(1)

    metadata_df = pd.read_csv(metadata_path)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process subject
    process_participant(args.subject, config, metadata_df, output_dir)


if __name__ == "__main__":
    main()
