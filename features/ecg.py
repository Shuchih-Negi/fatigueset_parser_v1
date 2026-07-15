"""
ECG/HRV feature extraction.

Pure functions to compute heart-rate and heart-rate-variability features
from R-R intervals and raw ECG waveforms. No file I/O.
"""

import numpy as np
from scipy import signal
from typing import Dict, Optional, Tuple


def compute_features(
    rr_intervals_ms: np.ndarray,
    ecg_waveform: Optional[np.ndarray] = None,
    invalid_value: int = 4095,
    sample_rate_hz: float = 250.0,
    include_lf_hf: bool = True,
) -> Dict[str, float]:
    """
    Compute HR and HRV features from R-R intervals and optional raw ECG.

    Args:
        rr_intervals_ms: array of R-R interval durations in milliseconds
        ecg_waveform: optional array of raw ECG samples (12-bit values).
                      Pass None if ECG data is unavailable for this window.
        invalid_value: sentinel value for invalid ECG samples (e.g., 4095 for 12-bit overflow)
        sample_rate_hz: ECG sampling rate in Hz (default 250 for Zephyr BioHarness)
        include_lf_hf: whether to compute LF/HF (default True; set False for 30s windows)

    Returns:
        dict with keys:
            - hr: heart rate in beats per minute
            - rmssd: root mean square of successive RR differences, in ms
            - sdnn: standard deviation of RR intervals, in ms
            - lf_hf: ratio of low-frequency to high-frequency power (NaN if not computed)
            - data_quality: "valid" if < 10% invalid samples, "artefact" if > 10%,
                            or "unknown" if ECG data was not available for this window
    """
    result = {
        "hr": np.nan,
        "rmssd": np.nan,
        "sdnn": np.nan,
        "lf_hf": np.nan,
        "data_quality": "unknown" if ecg_waveform is None else "valid",
    }

    # --- HR: beats per minute ---
    if len(rr_intervals_ms) > 0:
        mean_rr_ms = np.nanmean(rr_intervals_ms)
        if mean_rr_ms > 0:
            result["hr"] = 60000.0 / mean_rr_ms
        else:
            result["hr"] = np.nan

    # --- RMSSD: root mean square of successive differences ---
    if len(rr_intervals_ms) > 1:
        valid_rr = rr_intervals_ms[~np.isnan(rr_intervals_ms)]
        if len(valid_rr) > 1:
            successive_diff = np.diff(valid_rr)
            rmssd = np.sqrt(np.mean(successive_diff ** 2))
            result["rmssd"] = rmssd

    # --- SDNN: standard deviation of RR intervals ---
    if len(rr_intervals_ms) > 1:
        valid_rr = rr_intervals_ms[~np.isnan(rr_intervals_ms)]
        if len(valid_rr) > 1:
            result["sdnn"] = np.std(valid_rr)

    # --- LF/HF: low-frequency to high-frequency power ratio ---
    if include_lf_hf and len(rr_intervals_ms) > 1:
        valid_rr = rr_intervals_ms[~np.isnan(rr_intervals_ms)]
        if len(valid_rr) > 10:  # need minimum samples for FFT
            result["lf_hf"] = _compute_lf_hf_ratio(valid_rr)

    # --- Data quality: check for invalid ECG samples ---
    if ecg_waveform is not None and len(ecg_waveform) > 0:
        invalid_ratio = np.sum(ecg_waveform == invalid_value) / len(ecg_waveform)
        if invalid_ratio > 0.10:  # > 10% invalid
            result["data_quality"] = "artefact"

    return result


def compute_lf_hf(
    rr_intervals_ms: np.ndarray,
    min_samples: int = 10,
) -> Tuple[float, float, bool]:
    """
    Compute LF/HF ratio with confidence information.

    Args:
        rr_intervals_ms: array of RR intervals in milliseconds
        min_samples: minimum number of RR intervals needed for reliable computation

    Returns:
        Tuple of (lf_hf_ratio, actual_rr_duration_sec, low_confidence)
        - lf_hf_ratio: computed LF/HF ratio (NaN if not computable)
        - actual_rr_duration_sec: time span of RR data used (sum of RR intervals in seconds)
        - low_confidence: True if actual_rr_duration_sec < 90s (less than 90s of RR data)
    """
    if len(rr_intervals_ms) < min_samples:
        return np.nan, 0.0, True

    valid_rr = rr_intervals_ms[~np.isnan(rr_intervals_ms)]
    if len(valid_rr) < min_samples:
        return np.nan, 0.0, True

    # Compute actual RR duration covered
    actual_duration_sec = np.sum(valid_rr) / 1000.0

    # Low confidence if less than 90s of RR data (standard 2-min window is 120s)
    low_confidence = actual_duration_sec < 90.0

    lf_hf = _compute_lf_hf_ratio(valid_rr)

    return lf_hf, actual_duration_sec, low_confidence


def _compute_lf_hf_ratio(rr_intervals_ms: np.ndarray) -> float:
    """
    Compute LF/HF power ratio from RR intervals.

    Uses Welch's method to estimate power spectral density in the frequency bands:
    - LF (low frequency): 0.04–0.15 Hz
    - HF (high frequency): 0.15–0.4 Hz

    Args:
        rr_intervals_ms: array of RR intervals in milliseconds

    Returns:
        lf_hf: ratio of LF to HF power (or NaN if not computable)
    """
    try:
        # Convert RR intervals to instantaneous heart rate
        # This requires interpolation to get uniform sampling
        rr_seconds = rr_intervals_ms / 1000.0
        mean_rr = np.mean(rr_seconds)

        # Create time axis for RR intervals (cumulative)
        time_rr = np.cumsum(np.concatenate(([0], rr_seconds)))

        # Resample to 4 Hz (standard for HRV analysis)
        resample_rate = 4.0
        time_uniform = np.arange(time_rr[0], time_rr[-1], 1.0 / resample_rate)
        if len(time_uniform) < 10:
            return np.nan

        # Interpolate RR intervals to uniform grid
        hr_instantaneous = np.interp(time_uniform, time_rr[:-1], 60.0 / rr_seconds)

        # Compute power spectral density using Welch's method
        nperseg = min(256, len(hr_instantaneous) // 2)
        if nperseg < 16:
            return np.nan

        f, pxx = signal.welch(
            hr_instantaneous,
            fs=resample_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
        )

        # Define frequency bands
        lf_mask = (f >= 0.04) & (f < 0.15)
        hf_mask = (f >= 0.15) & (f < 0.4)

        lf_power = np.trapz(pxx[lf_mask], f[lf_mask])
        hf_power = np.trapz(pxx[hf_mask], f[hf_mask])

        if hf_power > 0:
            return lf_power / hf_power
        else:
            return np.nan

    except Exception:
        return np.nan
