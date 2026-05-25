"""
preprocessing.py
================
Preprocesses raw telemetry CSV into sliding window sequences with stat features.
Matches the logic from the offline training pipeline.

KEY FIX: If the stored scalers produce out-of-range outputs (because the
uploaded telemetry has a different scale than training data), we refit the
scalers on the current data. This allows the model to detect relative anomalies
within the uploaded dataset rather than failing with exploding MSE.
"""

import numpy as np
import pandas as pd
import pickle

from scipy.signal import find_peaks

from config import (
    MAX_LEN,
    N_FEATURES,
    WINDOW_STEP,
    STAT_FEATURES,
    SIGNAL_SCALER_PATH,
    STAT_SCALER_PATH
)


# =========================================================
# LOAD SCALERS
# =========================================================

def load_scalers():
    """Load pre-fitted scalers from offline training."""
    with open(SIGNAL_SCALER_PATH, "rb") as f:
        sig_scaler = pickle.load(f)
    with open(STAT_SCALER_PATH, "rb") as f:
        stat_scaler = pickle.load(f)
    return sig_scaler, stat_scaler


def _scalers_compatible(sig_scaler, values: np.ndarray) -> bool:
    """
    Check if the stored scaler is compatible with the current data.
    If the scaler produces values wildly outside [−5, 5] on the bulk
    of the data, the scaler was fitted on a different distribution.
    """
    try:
        sample = values[:min(500, len(values))].reshape(-1, 1)
        scaled = sig_scaler.transform(sample)
        # If median absolute scaled value > 10, scaler is incompatible
        return float(np.median(np.abs(scaled))) <= 10.0
    except Exception:
        return False


def _refit_scalers(values: np.ndarray, stat_matrix: np.ndarray):
    """
    Refit MinMaxScaler on the current data.
    Used when stored scalers are incompatible with uploaded telemetry.
    """
    from sklearn.preprocessing import MinMaxScaler
    sig_scaler  = MinMaxScaler()
    stat_scaler = MinMaxScaler()
    sig_scaler.fit(values.reshape(-1, 1))
    if stat_matrix is not None and len(stat_matrix) > 0:
        stat_scaler.fit(stat_matrix)
    print("[SCALER] Stored scalers incompatible — refitted on current data.")
    return sig_scaler, stat_scaler


# =========================================================
# STAT FEATURE EXTRACTION
# =========================================================

def compute_stat_features(signal: np.ndarray) -> np.ndarray:
    """
    Compute the 16 statistical features for a signal window.
    Must match training feature order exactly.
    """
    signal = signal.flatten()
    n = len(signal)

    if n == 0:
        return np.zeros(len(STAT_FEATURES), dtype=np.float32)

    mean_val = np.mean(signal)
    var_val  = np.var(signal)
    std_val  = np.std(signal)

    diff_from_mean = signal - mean_val
    m2 = np.mean(diff_from_mean ** 2)
    m3 = np.mean(diff_from_mean ** 3)
    m4 = np.mean(diff_from_mean ** 4)
    eps = 1e-8

    skew_val = m3 / (np.power(m2, 1.5) + eps)
    kurt_val = (m4 / (m2 ** 2 + eps)) - 3.0

    peaks, _  = find_peaks(signal)
    n_peaks   = len(peaks)

    sm10_peaks = 0
    if n >= 10:
        s10 = np.convolve(signal, np.ones(10) / 10, mode='valid')
        s10_peaks, _ = find_peaks(s10)
        sm10_peaks = len(s10_peaks)

    sm20_peaks = 0
    if n >= 20:
        s20 = np.convolve(signal, np.ones(20) / 20, mode='valid')
        s20_peaks, _ = find_peaks(s20)
        sm20_peaks = len(s20_peaks)

    diff1 = np.diff(signal)
    diff2 = np.diff(diff1) if len(diff1) > 0 else np.array([])

    d1_peaks, _   = find_peaks(diff1)
    d2_peaks, _   = find_peaks(diff2)
    diff_peaks_val  = len(d1_peaks)
    diff2_peaks_val = len(d2_peaks)
    diff_var_val    = np.var(diff1)  if len(diff1) > 0 else 0.0
    diff2_var_val   = np.var(diff2) if len(diff2) > 0 else 0.0

    gaps_sq      = 0.0
    len_weighted = float(n)
    var_div_dur  = var_val / (n + eps)
    var_div_len  = var_val / (n + eps)

    return np.array([
        mean_val, var_val, std_val,
        kurt_val, skew_val,
        n_peaks, sm10_peaks, sm20_peaks,
        diff_peaks_val, diff2_peaks_val,
        diff_var_val, diff2_var_val,
        gaps_sq, len_weighted,
        var_div_dur, var_div_len,
    ], dtype=np.float32)


# =========================================================
# MAIN TELEMETRY PREPROCESSING
# =========================================================

def process_telemetry_csv(df: pd.DataFrame) -> dict:
    """
    Convert raw telemetry CSV into sliding windows
    compatible with trained LSTM autoencoder.
    """

    # ── Column detection ──────────────────────────────────
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    timestamp_col = next(
        (c for c in cols if c in {"timestamp","time","datetime","date","ts","epoch"}), None)
    channel_col = next(
        (c for c in cols if c in {"channel","sensor","subsystem","parameter","signal"}), None)
    value_col = next(
        (c for c in cols if c in {"value","reading","current","voltage","temperature","sensor_value"}), None)

    if value_col is None:
        blacklist = {"segment","index","id","row","anomaly","train","channel","timestamp"}
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns
                        if c not in blacklist]
        if numeric_cols:
            value_col = numeric_cols[0]

    if channel_col is None:
        df["channel"] = "MAIN"
        channel_col   = "channel"

    if timestamp_col is None:
        raise ValueError("No timestamp column detected.")
    if value_col is None:
        raise ValueError("No telemetry value column detected.")

    df = df.rename(columns={
        timestamp_col: "timestamp",
        channel_col:   "channel",
        value_col:     "value",
    })

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(["channel", "timestamp"])

    # ── Load scalers ──────────────────────────────────────
    sig_scaler, stat_scaler = load_scalers()

    # ── Check compatibility on first pass (all values) ────
    all_values = df["value"].values.astype(np.float32)

    # Pre-compute all stat features to check stat_scaler too
    _stat_sample = []
    for _, grp in df.groupby("channel"):
        vals = grp["value"].values.astype(np.float32)
        if len(vals) >= MAX_LEN:
            _stat_sample.append(compute_stat_features(vals[:MAX_LEN]))
    stat_matrix = np.array(_stat_sample) if _stat_sample else None

    compatible = _scalers_compatible(sig_scaler, all_values)
    if not compatible:
        sig_scaler, stat_scaler = _refit_scalers(all_values, stat_matrix)

    # ── Process each channel ──────────────────────────────
    result = {}

    for ch_name, grp in df.groupby("channel"):
        times  = grp["timestamp"].values
        values = grp["value"].values.astype(np.float32)
        n_pts  = len(values)

        print(f"[PREPROCESS] Channel={ch_name} Points={n_pts}")

        # Pad short streams
        if n_pts < MAX_LEN:
            pad_len = MAX_LEN - n_pts
            values  = np.pad(values, (0, pad_len), mode="edge")
            times   = np.pad(times,  (0, pad_len), mode="edge")
            n_pts   = len(values)

        X_windows, t_windows, indices, segment_ids = [], [], [], []

        for idx, start in enumerate(range(0, n_pts - MAX_LEN + 1, WINDOW_STEP)):
            end        = start + MAX_LEN
            win_val    = values[start:end]
            win_t      = times[start:end]
            stats      = compute_stat_features(win_val)

            # Scale signal
            win_val_scaled = sig_scaler.transform(win_val.reshape(-1, 1))
            win_val_scaled = np.clip(win_val_scaled, -50.0, 50.0)

            # Scale stats
            stats_scaled = stat_scaler.transform(stats.reshape(1, -1))
            stats_scaled = np.clip(stats_scaled, -50.0, 50.0)

            stats_tiled = np.tile(stats_scaled, (MAX_LEN, 1))
            combined    = np.hstack([win_val_scaled, stats_tiled])  # (MAX_LEN, 17)

            X_windows.append(combined)
            t_windows.append(win_t)
            indices.append((start, end))
            segment_ids.append(idx + 1)

        print(f"[PREPROCESS] Generated {len(X_windows)} windows for {ch_name}")

        result[ch_name] = {
            "X":              np.array(X_windows, dtype=np.float32),
            "timestamps":     np.array(t_windows),
            "window_indices": indices,
            "segment_ids":    segment_ids,
            "raw_values":     values,
            "raw_times":      times,
        }

    return result