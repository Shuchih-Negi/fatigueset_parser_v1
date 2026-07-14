"""
Windowing utilities for time-series data.

Dataset-agnostic slicing of signals and event series into fixed-duration windows.
Handles both fixed-time windowing and task-aligned windowing (if markers provided).
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


def window_generator(
    timestamps: np.ndarray,
    signals_dict: Dict[str, np.ndarray],
    window_seconds: float,
    overlap_seconds: float = 0.0,
    task_markers: Optional[pd.DataFrame] = None,
) -> Iterator[Tuple[float, float, int, Dict[str, np.ndarray]]]:
    """
    Generate consecutive (or task-aligned) time windows from signals.

    Args:
        timestamps: array of timestamps (in seconds or milliseconds; must be consistent with signals)
        signals_dict: dict mapping signal name (str) to array of values, same length as timestamps
        window_seconds: window duration in seconds
        overlap_seconds: overlap between consecutive windows in seconds (0 for no overlap)
        task_markers: optional DataFrame with columns [utcTime, eventMarker] for task-aligned windowing.
                      If provided, windows are generated only within task block boundaries,
                      skipping inter-block gaps. If None, use fixed-time windowing over the full signal range.

    Yields:
        (window_start_ts, window_end_ts, block_index, data_for_window)
        where data_for_window is a dict with same keys as signals_dict,
        containing only samples within [window_start_ts, window_end_ts].
        block_index is -1 for fixed-time windowing, or the index of the task block
        (0, 1, 2, ...) for task-aligned windowing.
    """
    if len(timestamps) == 0:
        return

    timestamps = _normalize_timestamps(timestamps)

    window_step = window_seconds - overlap_seconds

    task_blocks = extract_task_blocks(task_markers) if task_markers is not None else []
    use_task = task_markers is not None and len(task_markers) > 0 and len(task_blocks) > 0

    if not use_task:
        window_start = timestamps.min()
        while window_start < timestamps.max():
            window_end = window_start + window_seconds
            mask = (timestamps >= window_start) & (timestamps < window_end)
            if mask.any():
                data_for_window = {
                    signal_name: signal_array[mask]
                    for signal_name, signal_array in signals_dict.items()
                }
                yield window_start, window_end, -1, data_for_window
            window_start += window_step
    else:
        # Task-aligned windowing: generate windows only within task blocks
        for block_idx, (block_start, block_end) in enumerate(task_blocks):
            window_start = block_start
            while window_start < block_end:
                window_end = min(window_start + window_seconds, block_end)
                mask = (timestamps >= window_start) & (timestamps < window_end)
                if mask.any():
                    data_for_window = {
                        signal_name: signal_array[mask]
                        for signal_name, signal_array in signals_dict.items()
                    }
                    yield window_start, window_end, block_idx, data_for_window
                window_start += window_step
