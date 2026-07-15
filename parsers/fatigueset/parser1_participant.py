"""
Parser 1: Process individual FatigueSet participant.

Loads ECG/RR interval data, applies windowing and feature extraction,
joins labels from exp_fatigue.csv and metadata, outputs one CSV per subject.

Window boundaries are defined purely by exp_markers.csv block boundaries
and config window width. No sensor file health affects window generation.
If a window has no usable ECG/RR data, it still appears in output with
data_quality = "unknown" and NaN features.

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features import ecg
from features.windowing import (
    extract_task_blocks,
    window_boundaries,
    lf_hf_window_boundaries,
    extract_signal_in_window,
    compute_rr_duration_in_window,
)

logger = logging.getLogger(__name__)


def build_intensity_lookup(
    subject_id: int, metadata_df: pd.DataFrame
) -> Dict[int, str]:
    row = metadata_df[metadata_df["participant_id"] == subject_id]
    if len(row) == 0:
        logger.warning(f"No metadata row found for subject {subject_id}")
        return {1: np.nan, 2: np.nan, 3: np.nan}
    row = row.iloc[0]
    return {
        int(row["low_session"]): "low",
        int(row["medium_session"]): "medium",
        int(row["high_session"]): "high",
    }


def load_exp_fatigue_lookup(subject_id: int, session_id: int, raw_root: Path) -> pd.DataFrame:
    fatigue_path = raw_root / f"{subject_id:02d}" / f"{session_id:02d}" / "exp_fatigue.csv"
    if not fatigue_path.exists():
        logger.warning(f"No exp_fatigue.csv found at {fatigue_path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(fatigue_path)
    except Exception as e:
        logger.warning(f"Failed to load {fatigue_path}: {e}")
        return pd.DataFrame()


def load_task_markers(subject_id: int, session_id: int, raw_root: Path) -> pd.DataFrame:
    markers_path = raw_root / f"{subject_id:02d}" / f"{session_id:02d}" / "exp_markers.csv"
    if not markers_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(markers_path)
    except Exception as e:
        logger.warning(f"Failed to load {markers_path}: {e}")
        return pd.DataFrame()


def get_fatigue_for_window(
    window_start_sec: float,
    fatigue_df: pd.DataFrame,
    timestamp_field: str = "fatigueSurveySubmissionTime",
    value_field: str = "physicalFatigueScore",
    tolerance_seconds: float = 5.0,
) -> float:
    if fatigue_df.empty or timestamp_field not in fatigue_df.columns:
        return np.nan
    timestamps_sec = fatigue_df[timestamp_field].values
    distances = np.abs(timestamps_sec - window_start_sec)
    closest_idx = np.argmin(distances)
    if distances[closest_idx] <= tolerance_seconds:
        return fatigue_df[value_field].iloc[closest_idx]
    return np.nan


def load_survey_data(subject_id, session_id, survey_path, sss_field, gvas_field, sss_map):
    if not survey_path.exists():
        return np.nan, np.nan
    try:
        df = pd.read_excel(survey_path)
        row = df[(df["ID"] == subject_id) & (df["Session"] == session_id)]
        if len(row) == 0:
            return np.nan, np.nan
        row = row.iloc[0]
        sss_text = row.get(sss_field, np.nan)
        sss_label = sss_map.get(sss_text, np.nan)
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
    raw_root = Path(config["raw_root"])
    window_seconds = config["window_seconds"]
    overlap_seconds = config["overlap_seconds"]
    invalid_ecg_value = config["invalid_ecg_value"]
    fatigue_tolerance = config["fatigue_timestamp_tolerance_seconds"]
    use_task_markers = config.get("use_task_markers", False)
    ecg_sample_rate_hz = config.get("ecg_sample_rate_hz", 250.0)

    intensity_lookup = build_intensity_lookup(subject_id, metadata_df)

    survey_path = raw_root / config["survey_file"]
    sss_map = config.get("sss_text_map", {})
    sss_field = config.get("survey_sss_field", "Please indicate your current level of sleepiness.")
    gvas_field = config.get("survey_gvas_field", "How sleepy do you feel?")

    all_rows = []

    for session_id in [1, 2, 3]:
        logger.info(f"Processing subject {subject_id}, session {session_id}")

        ecg_path = raw_root / f"{subject_id:02d}" / f"{session_id:02d}" / config["ecg_waveform_file"]
        rr_path = raw_root / f"{subject_id:02d}" / f"{session_id:02d}" / config["rr_interval_file"]

        if not ecg_path.exists() or not rr_path.exists():
            logger.error(f"Missing ECG or RR file for subject {subject_id}, session {session_id}")
            continue

        try:
            ecg_df = pd.read_csv(ecg_path)
            rr_df = pd.read_csv(rr_path)
        except Exception as e:
            logger.error(f"Failed to load ECG/RR data: {e}")
            continue

        fatigue_df = load_exp_fatigue_lookup(subject_id, session_id, raw_root)
        use_exp_fatigue = not fatigue_df.empty

        markers_df_raw = load_task_markers(subject_id, session_id, raw_root) if use_task_markers else pd.DataFrame()

        sss_pretask, gvas_pretask = load_survey_data(
            subject_id, session_id, survey_path, sss_field, gvas_field, sss_map
        )
        intensity_level = intensity_lookup.get(session_id, np.nan)

        try:
            ecg_waveform = ecg_df["ecg_waveform"].values
            rr_timestamps_ms = rr_df["timestamp"].values
            rr_durations = rr_df["duration"].values

            session_start_ms = rr_timestamps_ms.min()

            # Normalize RR timestamps to seconds from session start
            rr_timestamps_sec = (rr_timestamps_ms - session_start_ms) / 1000.0
            neg_rr = rr_timestamps_sec[rr_timestamps_sec < 0]
            if len(neg_rr) > 0:
                logger.warning(f"Clamping {len(neg_rr)} negative RR timestamps to 0 (min: {neg_rr.min():.6f}s)")
                rr_timestamps_sec = np.maximum(rr_timestamps_sec, 0.0)

            # Normalize ECG timestamps (may be corrupted)
            ecg_timestamps_ms = ecg_df["timestamp"].values
            ecg_timestamps_sec = (ecg_timestamps_ms - session_start_ms) / 1000.0
            ecg_valid = True
            if len(np.unique(ecg_timestamps_sec)) == 1:
                logger.warning(f"ECG timestamps corrupted for P{subject_id:02d} S{session_id} -- ECG quality will be unknown")
                ecg_valid = False
            else:
                neg_ecg = ecg_timestamps_sec[ecg_timestamps_sec < 0]
                if len(neg_ecg) > 0:
                    ecg_timestamps_sec = np.maximum(ecg_timestamps_sec, 0.0)

            # Normalize marker timestamps
            markers_df = markers_df_raw.copy() if not markers_df_raw.empty else pd.DataFrame()
            if not markers_df.empty:
                markers_df["utcTime"] = (markers_df["utcTime"].values.astype(float) - session_start_ms) / 1000.0
                neg_m = markers_df["utcTime"][markers_df["utcTime"] < 0]
                if len(neg_m) > 0:
                    markers_df["utcTime"] = np.maximum(markers_df["utcTime"], 0.0)
                logger.debug(f"Loaded {len(markers_df)} task markers for P{subject_id:02d} S{session_id}")

            # Signal time range (for fallback windowing when no markers)
            rr_start = rr_timestamps_sec.min()
            rr_end = rr_timestamps_sec.max()

            # Pre-compute block-level fatigue mapping
            block_fatigue_map = {}
            if use_exp_fatigue and use_task_markers and not markers_df.empty:
                blocks = extract_task_blocks(markers_df)
                fatigue_sorted = fatigue_df.sort_values(
                    config.get("exp_fatigue_timestamp_field", "fatigueSurveySubmissionTime")
                ).reset_index(drop=True)
                for i, (b_start, b_end) in enumerate(blocks):
                    if i < len(fatigue_sorted):
                        block_fatigue_map[i] = fatigue_sorted.iloc[i][
                            config.get("exp_fatigue_physical_field", "physicalFatigueScore")
                        ]
                    else:
                        block_fatigue_map[i] = np.nan
                logger.debug(f"Block-fatigue mapping: {block_fatigue_map}")

            # ============================================================
            # LF/HF computation: separate 2-minute windowing pass
            # ============================================================
            lf_hf_window_seconds = config.get("lf_hf_window_seconds", 120.0)
            lf_hf_min_confidence_sec = config.get("lf_hf_min_confidence_seconds", 90.0)

            lf_hf_results = []
            for lf_ws, lf_we, lf_block_idx in lf_hf_window_boundaries(
                task_markers=markers_df if use_task_markers else None,
                signal_start_sec=rr_start,
                signal_end_sec=rr_end,
                window_seconds=lf_hf_window_seconds,
            ):
                rr_in_lf = extract_signal_in_window(rr_timestamps_sec, rr_durations, lf_ws, lf_we)
                actual_dur = compute_rr_duration_in_window(rr_timestamps_sec, rr_durations, lf_ws, lf_we)
                if len(rr_in_lf) > 10:
                    lf_hf_val, lf_hf_dur, low_conf = ecg.compute_lf_hf(rr_in_lf)
                else:
                    lf_hf_val, lf_hf_dur, low_conf = np.nan, actual_dur, True
                lf_hf_results.append({
                    "start": lf_ws, "end": lf_we, "block_idx": lf_block_idx,
                    "lf_hf": lf_hf_val, "duration_sec": lf_hf_dur,
                    "low_confidence": low_conf,
                })

            # ============================================================
            # 30s window generation -- windows defined by markers only
            # ============================================================
            for window_start, window_end, block_idx in window_boundaries(
                task_markers=markers_df if use_task_markers else None,
                signal_start_sec=rr_start,
                signal_end_sec=rr_end,
                window_seconds=window_seconds,
                overlap_seconds=overlap_seconds,
            ):
                # Extract RR intervals for this window (using RR timestamps)
                rr_in_window = extract_signal_in_window(rr_timestamps_sec, rr_durations, window_start, window_end)

                # Extract ECG waveform for this window
                if ecg_valid:
                    ecg_in_window = extract_signal_in_window(ecg_timestamps_sec, ecg_waveform, window_start, window_end)
                else:
                    ecg_in_window = None

                # Compute features
                if len(rr_in_window) == 0:
                    # No RR data at all -- window exists but has no features
                    features = {
                        "hr": np.nan, "rmssd": np.nan, "sdnn": np.nan,
                        "lf_hf": np.nan, "data_quality": "unknown",
                    }
                else:
                    features = ecg.compute_features(
                        rr_in_window,
                        ecg_waveform=ecg_in_window,
                        invalid_value=invalid_ecg_value,
                        sample_rate_hz=ecg_sample_rate_hz,
                        include_lf_hf=False,
                    )
                    # Override data_quality if ECG was not available
                    if not ecg_valid:
                        features["data_quality"] = "unknown"

                # Map LF/HF from 2-minute window
                lf_hf_val = np.nan
                lf_hf_duration = np.nan
                lf_hf_low_conf = True
                for lf_res in lf_hf_results:
                    if window_start >= lf_res["start"] and window_end <= lf_res["end"]:
                        if use_task_markers and block_idx >= 0:
                            if lf_res["block_idx"] == block_idx:
                                lf_hf_val = lf_res["lf_hf"]
                                lf_hf_duration = lf_res["duration_sec"]
                                lf_hf_low_conf = lf_res["low_confidence"]
                                break
                        else:
                            lf_hf_val = lf_res["lf_hf"]
                            lf_hf_duration = lf_res["duration_sec"]
                            lf_hf_low_conf = lf_res["low_confidence"]
                            break

                # Determine label
                if use_exp_fatigue:
                    if use_task_markers and block_idx >= 0 and block_idx in block_fatigue_map:
                        window_label = block_fatigue_map[block_idx]
                    else:
                        window_label = get_fatigue_for_window(
                            window_start, fatigue_df,
                            timestamp_field=config.get("exp_fatigue_timestamp_field", "fatigueSurveySubmissionTime"),
                            value_field=config.get("exp_fatigue_physical_field", "physicalFatigueScore"),
                            tolerance_seconds=fatigue_tolerance,
                        )
                    window_label_type = config.get("label_type", "fatigue_rating")
                else:
                    if not pd.isna(sss_pretask):
                        logger.warning(
                            f"exp_fatigue.csv missing for P{subject_id:02d} S{session_id} -- "
                            f"falling back to SSS pretask label"
                        )
                    window_label = sss_pretask
                    window_label_type = "sss_pretask"

                row = {
                    "subject_id": f"P{subject_id:02d}",
                    "session_id": session_id,
                    "window_start_sec": window_start,
                    "hr": features["hr"],
                    "rmssd": features["rmssd"],
                    "sdnn": features["sdnn"],
                    "lf_hf": lf_hf_val,
                    "lf_hf_window_sec": lf_hf_duration,
                    "lf_hf_low_confidence": lf_hf_low_conf,
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
            import traceback
            traceback.print_exc()
            continue

    if len(all_rows) == 0:
        logger.error(f"No data for subject {subject_id}")
        return

    output_df = pd.DataFrame(all_rows)
    output_path = output_dir / f"P{subject_id:02d}.csv"
    output_df.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(output_df)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process one FatigueSet participant and output per-subject CSV."
    )
    parser.add_argument("--subject", type=int, required=True, help="Subject ID (1-12)")
    parser.add_argument("--config", type=str, default="configs/datasets/fatigueset.yaml")
    parser.add_argument("--output-dir", type=str, default="raw/fatigueset_participants")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    metadata_path = Path(config["raw_root"]) / config["metadata_file"]
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        sys.exit(1)

    metadata_df = pd.read_csv(metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    process_participant(args.subject, config, metadata_df, output_dir)


if __name__ == "__main__":
    main()
