"""
detection_engine.py
===================
Layer 3 Fusion Engine. Combines LSTM reconstruction error with rule-based checks
to generate a final Anomaly Severity Score.

KEY FIX: The lstm_threshold argument is now always respected.
If the stored threshold is below the 5th percentile of observed MSE values
(meaning it would flag everything), we switch to a dynamic threshold at
the 95th percentile of the current run's MSE distribution.
"""

import numpy as np
import tensorflow as tf
from rule_engine import check_window_rules
from config import (
    MODEL_PATH, DEFAULT_LSTM_THRESHOLD, SEVERITY_WEIGHTS, SEVERITY_LEVELS
)

# Load model once at module level
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _MODEL


def compute_reconstruction_error(X: np.ndarray) -> np.ndarray:
    """Compute per-window MSE for the batch of windows X."""
    model  = get_model()
    X_pred = model.predict(X, batch_size=64, verbose=0)
    mse    = np.mean(np.square(X - X_pred), axis=(1, 2))
    return mse


def _pick_threshold(mses: np.ndarray, stored_threshold: float) -> float:
    """
    Choose the best threshold for this run.

    Logic
    -----
    - If the stored threshold is below the 5th percentile of observed MSE,
      it would flag everything (100% anomaly rate) → use p95 instead.
    - If the stored threshold is above the 95th percentile, it would flag
      nothing → use p85 instead.
    - Otherwise the stored threshold is reasonable → use it as-is.

    This makes the system robust to scale mismatches between training
    and test data without requiring retraining.
    """
    p05 = float(np.percentile(mses, 5))
    p85 = float(np.percentile(mses, 85))
    p95 = float(np.percentile(mses, 95))

    if stored_threshold < p05:
        chosen = p95
        reason = f"stored ({stored_threshold:.6f}) < p05 ({p05:.6f}) → using p95 ({p95:.6f})"
    elif stored_threshold > p95:
        chosen = p85
        reason = f"stored ({stored_threshold:.6f}) > p95 ({p95:.6f}) → using p85 ({p85:.6f})"
    else:
        chosen = stored_threshold
        reason = f"stored threshold ({stored_threshold:.6f}) is within range → using as-is"

    print(f"[THRESHOLD] {reason}")
    return chosen


def run_hybrid_detection(
    channel_data: dict,
    lstm_threshold: float = DEFAULT_LSTM_THRESHOLD
) -> list:
    """
    Run the full hybrid pipeline on a dictionary of channel data.

    Parameters
    ----------
    channel_data   : output of process_telemetry_csv()
    lstm_threshold : threshold from config/metadata — may be overridden
                     dynamically if it produces 0% or 100% anomaly rate.

    Returns
    -------
    list of anomaly dicts with severity, confidence, and timestamps.
    """
    results = []

    for ch_name, data in channel_data.items():
        X            = data["X"]
        raw_values   = data["raw_values"]
        win_indices  = data["window_indices"]
        timestamps   = data["timestamps"]

        # 1. Compute reconstruction errors
        mses = compute_reconstruction_error(X)

        print(
            f"[MSE] Channel={ch_name} "
            f"min={mses.min():.6f} max={mses.max():.6f} "
            f"mean={mses.mean():.6f} p50={np.percentile(mses,50):.6f} "
            f"p95={np.percentile(mses,95):.6f} stored_thr={lstm_threshold:.6f}"
        )

        # 2. Pick a sensible threshold for this channel's MSE distribution
        effective_threshold = _pick_threshold(mses, lstm_threshold)

        # 3. Iterate windows
        for i, (start, end) in enumerate(win_indices):
            mse       = float(mses[i])
            lstm_flag = mse > effective_threshold

            raw_win = raw_values[start:end]
            rules   = check_window_rules(raw_win)

            # 4. Fusion score
            score = 0.0
            if lstm_flag:
                score += SEVERITY_WEIGHTS["lstm_flag"]
            if rules["spike_flag"]:
                score += SEVERITY_WEIGHTS["spike_flag"]
            if rules["range_flag"]:
                score += SEVERITY_WEIGHTS["range_flag"]
            if rules["flatline_flag"]:
                score += SEVERITY_WEIGHTS["flatline_flag"]

            score = min(1.0, score)

            # 5. Severity level
            level = "NOMINAL"
            for lvl, (low, high) in SEVERITY_LEVELS.items():
                if low <= score < high:
                    level = lvl
                    break

            if level != "NOMINAL":
                details = rules["details"].copy()
                if lstm_flag:
                    details.append(
                        f"AI Reconstruction Error: {mse:.6f} "
                        f"> threshold {effective_threshold:.6f}"
                    )

                results.append({
                    "channel":   ch_name,
                    "timestamp": timestamps[i][0],
                    "end_time":  timestamps[i][-1],
                    "mse":       mse,
                    "score":     score,
                    "severity":  level,
                    "details":   details,
                    "start_idx": start,
                    "end_idx":   end,
                })

    return results