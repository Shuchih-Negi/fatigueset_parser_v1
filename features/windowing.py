"""
Windowing utilities for time-series data.

Dataset-agnostic slicing of signals and event series into fixed-duration windows.
Handles both fixed-time windowing and task-aligned windowing (if markers provided).

Window boundaries are defined purely by markers + config width. No sensor file's
health should be able to zero out a session's window count. Data extraction into
windows happens outside this module.
"""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Dict, Optional, List


def _normalize_timestamps(timestamps: np.ndarray) -> np.ndarray:
    """Convert milliseconds to seconds if timestamps are in ms (> 1e9)."""
    if len(timestamps) == 0:
        return timestamps
    ts_min, ts_max = timestamps.min(), timestamps.max()
    if ts_min > 1e9:
        timestamps = timestamps / 1000.0
    return timestamps


def extract_task_blocks(
    markers_df: pd.DataFrame,
) -> List[Tuple[float, float]]:
    """
    Extract task block boundaries from marker DataFrame.

    Pairs start_* markers with their corresponding end_* markers.
    Blocks are returned in chronological order.
    Returns blocks as [(block_start_sec, block_end_sec), ...].

    Args:
        markers_df: DataFrame with columns [utcTime, eventMarker].
                    Timestamps should already be in seconds.

    Returns:
        List of (block_start, block_end) tuples in chronological order.
        Empty list if no valid start/end pairs found.
    """
    if markers_df.empty or "eventMarker" not in markers_df.columns:
        return []

    marker_ts = _normalize_timestamps(markers_df["utcTime"].values.astype(float))
    markers = markers_df["eventMarker"].values

    start_markers = {}
    blocks = []

    for ts, marker in zip(marker_ts, markers):
        if pd.isna(ts) or pd.isna(marker):
            continue
        if marker.startswith("start_"):
            block_name = marker[len("start_"):]
            start_markers[block_name] = ts
        elif marker.startswith("end_"):
            block_name = marker[len("end_"):]
            if block_name in start_markers:
                block_start = start_markers.pop(block_name)
                blocks.append((block_start, ts))

    blocks.sort(key=lambda x: x[0])
    return blocks


def window_boundaries(
    task_markers: Optional[pd.DataFrame] = None,
    signal_start_sec: float = 0.0,
    signal_end_sec: float = 0.0,
    window_seconds: float = 30.0,
    overlap_seconds: float = 0.0,
) -> Iterator[Tuple[float, float, int]]:
    """
    Generate window boundaries from markers or full signal range.

    Windows exist purely because markers say a block exists and config says
    how wide to slice it. No sensor file health affects window generation.

    Args:
        task_markers: DataFrame with [utcTime, eventMarker] for task-aligned windowing.
                      If provided and valid, windows are generated within block boundaries.
        signal_start_sec: start of signal range (used only when task_markers is None)
        signal_end_sec: end of signal range (used only when task_markers is None)
        window_seconds: window duration in seconds
        overlap_seconds: overlap between consecutive windows (0 for non-overlapping)

    Yields:
        (window_start_sec, window_end_sec, block_index)
        block_index is -1 for fixed-time, or block ordinal (0, 1, 2, ...) for task-aligned.
    """
    window_step = window_seconds - overlap_seconds

    task_blocks = extract_task_blocks(task_markers) if task_markers is not None else []
    use_task = task_markers is not None and len(task_markers) > 0 and len(task_blocks) > 0

    if not use_task:
        window_start = signal_start_sec
        while window_start + window_seconds <= signal_end_sec:
            window_end = window_start + window_seconds
            yield (window_start, window_end, -1)
            window_start += window_step
    else:
        for block_idx, (block_start, block_end) in enumerate(task_blocks):
            window_start = block_start
            while window_start + window_seconds <= block_end:
                window_end = window_start + window_seconds
                yield (window_start, window_end, block_idx)
                window_start += window_step


def lf_hf_window_boundaries(
    task_markers: Optional[pd.DataFrame] = None,
    signal_start_sec: float = 0.0,
    signal_end_sec: float = 0.0,
    window_seconds: float = 120.0,
) -> Iterator[Tuple[float, float, int]]:
    """
    Generate 2-minute LF/HF window boundaries from markers or full signal range.

    Args:
        task_markers: DataFrame with [utcTime, eventMarker]
        signal_start_sec: start of signal range
        signal_end_sec: end of signal range
        window_seconds: LF/HF window duration (default 120s)

    Yields:
        (window_start_sec, window_end_sec, block_index)
    """
    task_blocks = extract_task_blocks(task_markers) if task_markers is not None else []
    use_task = task_markers is not None and len(task_markers) > 0 and len(task_blocks) > 0

    if not use_task:
        window_start = signal_start_sec
        while window_start + window_seconds <= signal_end_sec:
            yield (window_start, window_start + window_seconds, -1)
            window_start += window_seconds
    else:
        for block_idx, (block_start, block_end) in enumerate(task_blocks):
            window_start = block_start
            while window_start + window_seconds <= block_end:
                yield (window_start, window_start + window_seconds, block_idx)
                window_start += window_seconds


def extract_signal_in_window(
    timestamps_sec: np.ndarray,
    signal: np.ndarray,
    window_start: float,
    window_end: float,
) -> np.ndarray:
    """
    Extract signal values that fall within a time window.

    Args:
        timestamps_sec: array of timestamps in seconds (normalized to session start)
        signal: array of values, same length as timestamps_sec
        window_start: window start time in seconds
        window_end: window end time in seconds

    Returns:
        Array of signal values within [window_start, window_end)
    """
    mask = (timestamps_sec >= window_start) & (timestamps_sec < window_end)
    return signal[mask]


def compute_rr_duration_in_window(
    rr_timestamps_sec: np.ndarray,
    rr_durations: np.ndarray,
    window_start: float,
    window_end: float,
) -> float:
    """
    Compute total duration of RR data present in a window.

    Args:
        rr_timestamps_sec: RR timestamps in seconds
        rr_durations: RR interval durations in milliseconds
        window_start: window start in seconds
        window_end: window end in seconds

    Returns:
        Total RR data duration in seconds (sum of RR intervals within window)
    """
    mask = (rr_timestamps_sec >= window_start) & (rr_timestamps_sec < window_end)
    rr_in_window = rr_durations[mask]
    if len(rr_in_window) > 0:
        return np.sum(rr_in_window) / 1000.0
    return 0.0
