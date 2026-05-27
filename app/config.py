"""
config.py
=========
Central configuration for the Telemetry Anomaly Intelligence Platform.
All thresholds, paths, and UI constants live here so they are easy to tune.
"""
import os
from pathlib import Path
# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "saved_model"))
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))

MODEL_PATH = MODEL_DIR / "lstm_autoencoder.keras"

SIGNAL_SCALER_PATH = MODEL_DIR / "signal_scaler.pkl"

STAT_SCALER_PATH = MODEL_DIR / "stat_scaler.pkl"

METADATA_PATH = MODEL_DIR / "metadata.json"

# ── Model hyperparameters (must match training) ───────────────────────────────
import json
# Defaults
MAX_LEN    = 100      # sliding window length (timesteps)
N_FEATURES = 17       # 1 signal + 16 stat features
LATENT_DIM = 64       # LSTM bottleneck units
DEFAULT_LSTM_THRESHOLD = 10.0   # default from training (87th percentile)


# Load dynamically from metadata.json if available
if METADATA_PATH.exists():
    try:
        with open(METADATA_PATH, "r") as f:
            _meta = json.load(f)
            DEFAULT_LSTM_THRESHOLD = _meta.get("threshold", DEFAULT_LSTM_THRESHOLD)
            MAX_LEN = _meta.get("max_len", MAX_LEN)
            LATENT_DIM = _meta.get("latent_dim", LATENT_DIM)
            print(f"[CONFIG] Loaded dynamic params from metadata: threshold={DEFAULT_LSTM_THRESHOLD:.6f}, max_len={MAX_LEN}")
    except Exception as e:
        print(f"[CONFIG] Failed to load dynamic metadata: {e}")
WINDOW_STEP = 10      # stride for sliding window (smaller = finer resolution)


# ── Statistical features computed per window (must match training order) ──────
STAT_FEATURES = [
    "mean", "var", "std", "kurtosis", "skew",
    "n_peaks", "smooth10_n_peaks", "smooth20_n_peaks",
    "diff_peaks", "diff2_peaks",
    "diff_var", "diff2_var",
    "gaps_squared", "len_weighted",
    "var_div_duration", "var_div_len",
]

# ── Detection thresholds ──────────────────────────────────────────────────────
SPIKE_ZSCORE_THRESHOLD   = 3.5        # z-score to flag a spike
FLATLINE_MAX_STD         = 1e-6       # std below this → stuck sensor
FLATLINE_MIN_WINDOW      = 20         # consecutive flat points to trigger
RATE_OF_CHANGE_THRESHOLD = 0.30       # 30% sudden change to flag


# ── Severity scoring weights ──────────────────────────────────────────────────
SEVERITY_WEIGHTS = {
    "lstm_flag"       : 0.45,   # AI reconstruction error
    "spike_flag"      : 0.25,   # sudden spike
    "range_flag"      : 0.15,   # out-of-range value
    "flatline_flag"   : 0.10,   # stuck sensor
    "stat_flag"       : 0.05,   # rolling statistical anomaly
}
# Severity thresholds (composite score 0-1)
SEVERITY_LEVELS = {
    "NOMINAL"   : (0.00, 0.20),
    "CAUTION"   : (0.20, 0.40),
    "WARNING"   : (0.40, 0.65),
    "CRITICAL"  : (0.65, 1.01),
}



# ── UI / theme constants ───────────────────────────────────────────────────────
PAGE_TITLE  = "NETRA — Telemetry Intelligence Platform"
PAGE_ICON   = ""

COLORS = {
    "bg_primary"   : "#000000",
    "bg_card"      : "#070707",
    "bg_panel"     : "#0b0b0b",
    "accent_cyan"  : "#dbeafe",
    "accent_blue"  : "#c7d2fe",
    "nominal"      : "#9ee6a7",
    "caution"      : "#ffe98a",
    "warning"      : "#ffb86b",
    "critical"     : "#ff6b6b",
    "text_primary" : "#e6eef6",
    "text_muted"   : "#94a3b8",
    "grid"         : "#111111",
}
CHANNEL_COLORS = [
    "#00d4ff", "#3b82f6", "#8b5cf6", "#ec4899",
    "#f59e0b", "#10b981", "#ef4444", "#6366f1",
    "#14b8a6", "#f97316",
]