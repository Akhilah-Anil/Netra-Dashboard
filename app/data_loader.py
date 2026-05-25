"""
data_loader.py
==============
Loads segments.csv + dataset.csv and produces padded per-segment tensors
for the LSTM Autoencoder.

Key facts confirmed from EDA
-----------------------------
- 2,123 (channel, segment) pairs — perfect 1-to-1 across both files
- Every segment is fully homogeneous (all rows same anomaly label)
- Segment lengths: min=8, max=1040, median~70, p95=464
- Train split: 1,594 train / 529 test  (via train column)
- Normal train segments: 1,273  |  Anomaly train: 321
- 9 channels, sampling values 1 or 5
- No NaN / inf in stat features

Tensor construction
-------------------
For each segment:
  - raw `value` time-series → shape (max_len, 1)   padded/truncated
  - 16 stat features from dataset.csv → replicated → (max_len, 16)
  - combined input: (max_len, 17)

Training uses ONLY normal segments (train==1 AND anomaly==0).
All-train segments (normal + anomaly) are built separately for threshold tuning.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

STAT_FEATURES = [
    'mean', 'var', 'std', 'kurtosis', 'skew',
    'n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks',
    'diff_peaks', 'diff2_peaks',
    'diff_var', 'diff2_var',
    'gaps_squared', 'len_weighted',
    'var_div_duration', 'var_div_len',
]

# ── Scaler range guard ────────────────────────────────────────────────────────
# If more than this fraction of transformed values fall outside [LO, HI],
# the training scaler is considered mismatched and we re-fit locally.
_RANGE_LO      = -0.5
_RANGE_HI      = 1.5
_OOB_THRESHOLD = 0.20   # 20 % out-of-bounds → re-fit


def _scaler_is_compatible(scaler: MinMaxScaler, data: np.ndarray) -> bool:
    """Return True if the scaler produces mostly in-range [−0.5, 1.5] outputs."""
    transformed = scaler.transform(data)
    oob = np.mean((transformed < _RANGE_LO) | (transformed > _RANGE_HI))
    if oob > _OOB_THRESHOLD:
        print(f"[SCALER GUARD] {oob*100:.1f}% values out of range — "
              f"training scaler is mismatched for this data.")
        return False
    return True


def _get_compatible_scalers(
        signal_data: np.ndarray,
        stat_data: np.ndarray,
        signal_scaler: MinMaxScaler,
        stat_scaler: MinMaxScaler,
) -> tuple[MinMaxScaler, MinMaxScaler, bool]:
    """
    Return scalers that are compatible with the supplied data.

    If the training scalers are compatible, return them unchanged (refitted=False).
    Otherwise create new scalers fitted on the local data (refitted=True).
    The caller can use `refitted` to decide whether to adjust the threshold.
    """
    sig_ok  = _scaler_is_compatible(signal_scaler, signal_data)
    stat_ok = _scaler_is_compatible(stat_scaler,   stat_data)

    if sig_ok and stat_ok:
        return signal_scaler, stat_scaler, False

    print("[SCALER GUARD] Re-fitting scalers on local test data. "
          "MSE scores will be relative to this data's own scale. "
          "Threshold will be adjusted automatically.")

    new_sig  = MinMaxScaler()
    new_stat = MinMaxScaler()
    new_sig.fit(signal_data)
    new_stat.fit(stat_data)
    return new_sig, new_stat, True


def load_raw(segments_path: str, dataset_path: str):
    """Load both CSVs, verify key alignment, return cleaned DataFrames."""
    seg = pd.read_csv(segments_path)
    ds  = pd.read_csv(dataset_path)

    seg_keys = set(zip(seg['channel'], seg['segment']))
    ds_keys  = set(zip(ds['channel'],  ds['segment']))
    diff = seg_keys.symmetric_difference(ds_keys)
    if diff:
        raise ValueError(
            f"Key mismatch between files — "
            f"{len(diff)} mismatched (channel, segment) pairs"
        )

    print(f"[DATA] segments.csv : {seg.shape[0]:,} rows | "
          f"{seg['channel'].nunique()} channels | "
          f"{seg['segment'].nunique()} segments")
    print(f"[DATA] dataset.csv  : {ds.shape[0]} rows (one per segment)")
    return seg, ds


def _build_tensors(seg_df: pd.DataFrame,
                   ds_df: pd.DataFrame,
                   signal_scaler: MinMaxScaler,
                   stat_scaler: MinMaxScaler,
                   max_len: int,
                   fit: bool = False):
    """
    Build (N, max_len, 17) tensor for a given subset of segments.

    Parameters
    ----------
    seg_df        : rows from segments.csv for this subset
    ds_df         : rows from dataset.csv for this subset (one per segment)
    signal_scaler : MinMaxScaler for the raw signal (channel 0)
    stat_scaler   : MinMaxScaler for the 16 stat features
    max_len       : fixed sequence length (pad zeros / truncate)
    fit           : if True, fit scalers before transforming (training only)

    Returns
    -------
    X        : np.ndarray (N, max_len, 17)
    meta_df  : pd.DataFrame with channel, segment, anomaly, train per segment
    refitted : bool — True when local scalers were used instead of training ones
    """
    stat_lookup = ds_df.set_index(['channel', 'segment'])[STAT_FEATURES]

    tensors = []
    meta    = []

    for (ch, sid), grp in seg_df.groupby(['channel', 'segment'], sort=False):
        grp    = grp.sort_values('timestamp') if 'timestamp' in grp.columns else grp
        signal = grp['value'].values.astype(np.float32).reshape(-1, 1)
        T      = len(signal)

        if T >= max_len:
            signal_padded = signal[:max_len]
        else:
            signal_padded = np.vstack(
                [signal, np.zeros((max_len - T, 1), dtype=np.float32)]
            )

        stats       = stat_lookup.loc[(ch, sid)].values.astype(np.float32)
        stats_tiled = np.tile(stats, (max_len, 1))

        combined = np.hstack([signal_padded, stats_tiled])
        tensors.append(combined)

        row = ds_df[
            (ds_df['channel'] == ch) & (ds_df['segment'] == sid)
        ].iloc[0]
        meta.append({
            'channel': ch,
            'segment': int(sid),
            'anomaly': int(row['anomaly']),
            'train':   int(row['train']),
        })

    X       = np.array(tensors, dtype=np.float32)
    meta_df = pd.DataFrame(meta)

    N, L, F = X.shape
    flat    = X.reshape(-1, F)

    refitted = False
    if fit:
        signal_scaler.fit(flat[:, :1])
        stat_scaler.fit(flat[:, 1:])
    else:
        # ── Scaler guard: re-fit locally if training scalers are mismatched ──
        signal_scaler, stat_scaler, refitted = _get_compatible_scalers(
            flat[:, :1], flat[:, 1:], signal_scaler, stat_scaler
        )

    flat[:, :1] = signal_scaler.transform(flat[:, :1])
    flat[:, 1:] = stat_scaler.transform(flat[:, 1:])
    X = flat.reshape(N, L, F)

    return X, meta_df, refitted


def prepare_training_data(segments_path: str,
                          dataset_path: str,
                          max_len: int = 100,
                          val_frac: float = 0.10,
                          save_dir: str = 'saved_model',
                          seed: int = 42):
    """
    Full training pipeline.

    Returns
    -------
    X_train, X_val, X_all_train, meta_all_train, signal_scaler, stat_scaler
    """
    os.makedirs(save_dir, exist_ok=True)

    seg, ds = load_raw(segments_path, dataset_path)

    seg_train = seg[seg['train'] == 1].copy()
    ds_train  = ds[ds['train']  == 1].copy()

    print(f"\n[DATA] Train segments (total): {ds_train.shape[0]} "
          f"| normal={int((ds_train['anomaly']==0).sum())} "
          f"| anomaly={int((ds_train['anomaly']==1).sum())}")

    ds_train_normal  = ds_train[ds_train['anomaly'] == 0].copy()
    normal_keys      = set(zip(ds_train_normal['channel'], ds_train_normal['segment']))
    seg_train_normal = seg_train[
        seg_train.apply(
            lambda r: (r['channel'], r['segment']) in normal_keys, axis=1
        )
    ].copy()

    print(f"[DATA] Normal training segments : {ds_train_normal.shape[0]}")
    print(f"[DATA] Normal training rows     : {len(seg_train_normal):,}")

    signal_scaler = MinMaxScaler()
    stat_scaler   = MinMaxScaler()

    X_normal, meta_normal, _ = _build_tensors(
        seg_train_normal, ds_train_normal,
        signal_scaler, stat_scaler,
        max_len, fit=True,
    )
    print(f"[DATA] X_normal shape : {X_normal.shape}  (fitted scalers on this)")

    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X_normal))
    n_val = max(1, int(len(X_normal) * val_frac))
    X_val   = X_normal[idx[:n_val]]
    X_train = X_normal[idx[n_val:]]
    print(f"[DATA] X_train : {X_train.shape}  X_val : {X_val.shape}")

    X_all_train, meta_all_train, _ = _build_tensors(
        seg_train, ds_train,
        signal_scaler, stat_scaler,
        max_len, fit=False,
    )
    print(f"[DATA] X_all_train (threshold tuning) : {X_all_train.shape}")

    for name, scaler in [('signal_scaler', signal_scaler),
                         ('stat_scaler',   stat_scaler)]:
        path = os.path.join(save_dir, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)
    print(f"[DATA] Scalers saved → {save_dir}/")

    return X_train, X_val, X_all_train, meta_all_train, signal_scaler, stat_scaler


def prepare_test_data(segments_path: str,
                      dataset_path: str,
                      signal_scaler: MinMaxScaler,
                      stat_scaler: MinMaxScaler,
                      max_len: int = 100):
    """
    Prepare test data using already-fitted scalers.
    Falls back to local re-fit if the training scalers are mismatched.

    Returns
    -------
    X_test    : (N_test, max_len, 17)
    meta_test : DataFrame with channel, segment, anomaly, train
    refitted  : bool — True when local scalers were used
    """
    seg, ds = load_raw(segments_path, dataset_path)

    if 'train' in seg.columns and seg['train'].nunique() > 1:
        seg_test = seg[seg['train'] == 0].copy()
        ds_test  = ds[ds['train']  == 0].copy()
    else:
        seg_test = seg.copy()
        ds_test  = ds.copy()
        if 'anomaly' not in ds_test.columns:
            ds_test['anomaly'] = -1
        if 'train' not in ds_test.columns:
            ds_test['train'] = 0

    print(f"[DATA] Test segments : {ds_test.shape[0]} "
          f"| anomaly={int((ds_test['anomaly']==1).sum())}"
          f"| normal={int((ds_test['anomaly']==0).sum())}")

    X_test, meta_test, refitted = _build_tensors(
        seg_test, ds_test,
        signal_scaler, stat_scaler,
        max_len, fit=False,
    )
    print(f"[DATA] X_test shape : {X_test.shape} | scaler_refitted={refitted}")
    return X_test, meta_test, refitted


# ── Stat feature helpers ──────────────────────────────────────────────────────

from scipy.signal import find_peaks


def compute_stat_features(signal: np.ndarray) -> np.ndarray:
    """Compute the 16 statistical features for a single 1D signal window."""
    signal = signal.flatten()
    n = len(signal)
    if n == 0:
        return np.zeros(16, dtype=np.float32)

    mean_val = np.mean(signal)
    var_val  = np.var(signal)
    std_val  = np.std(signal)

    diff_from_mean = signal - mean_val
    m2 = np.mean(diff_from_mean**2)
    m3 = np.mean(diff_from_mean**3)
    m4 = np.mean(diff_from_mean**4)

    eps      = 1e-8
    skew_val = m3 / (np.power(m2, 1.5) + eps)
    kurt_val = (m4 / (m2**2 + eps)) - 3.0

    peaks, _ = find_peaks(signal)
    n_peaks  = len(peaks)

    if n >= 10:
        s10        = np.convolve(signal, np.ones(10)/10, mode='valid')
        s10_peaks, _ = find_peaks(s10)
        sm10_peaks = len(s10_peaks)
    else:
        sm10_peaks = 0

    if n >= 20:
        s20        = np.convolve(signal, np.ones(20)/20, mode='valid')
        s20_peaks, _ = find_peaks(s20)
        sm20_peaks = len(s20_peaks)
    else:
        sm20_peaks = 0

    diff1  = np.diff(signal)
    diff2  = np.diff(diff1) if len(diff1) > 0 else np.array([])

    d1_peaks, _      = find_peaks(diff1)
    diff_peaks_val   = len(d1_peaks)
    d2_peaks, _      = find_peaks(diff2)
    diff2_peaks_val  = len(d2_peaks)

    diff_var_val  = np.var(diff1) if len(diff1) > 0 else 0
    diff2_var_val = np.var(diff2) if len(diff2) > 0 else 0

    gaps_sq      = 0.0
    len_weighted = n * 1.0
    var_div_dur  = var_val / (n + eps)
    var_div_len  = var_val / (n + eps)

    return np.array([
        mean_val, var_val, std_val, kurt_val, skew_val,
        n_peaks, sm10_peaks, sm20_peaks,
        diff_peaks_val, diff2_peaks_val,
        diff_var_val, diff2_var_val,
        gaps_sq, len_weighted,
        var_div_dur, var_div_len,
    ], dtype=np.float32)


def map_telemetry_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detects and renames synonyms of timestamp, value, and channel."""
    df = df.copy()
    col_mapping = {}

    for col in df.columns:
        col_lower = str(col).lower()
        if any(k in col_lower for k in ['time', 'date', 'ts', 'timestamp']):
            col_mapping[col] = 'timestamp'
        elif any(k in col_lower for k in ['value', 'val', 'current', 'voltage', 'reading']):
            col_mapping[col] = 'value'
        elif any(k in col_lower for k in ['channel', 'chan', 'ch']):
            col_mapping[col] = 'channel'

    df = df.rename(columns=col_mapping)

    if 'timestamp' not in df.columns:
        df['timestamp'] = np.arange(len(df))
    if 'value' not in df.columns:
        raise ValueError("No value or reading column found in telemetry file.")
    if 'channel' not in df.columns:
        df['channel'] = 'CH_001'

    return df


# ── Segment-style CSV ingestion (with scaler guard) ───────────────────────────

def prepare_segments_from_csv(df, signal_scaler, stat_scaler, max_len=100):
    """
    Build tensors from a single uploaded segments-style CSV.
    Auto-computes 16 stat features per (channel, segment) pair.

    Returns
    -------
    X        : (N, max_len, 17)
    meta_df  : DataFrame
    refitted : bool
    """
    from scipy import signal as scipy_signal
    from scipy.stats import kurtosis, skew as scipy_skew

    def _compute(vals, sampling=5):
        n        = len(vals)
        duration = n * sampling
        d        = np.diff(vals)
        d2       = np.diff(d) if len(d) > 1 else np.array([0.0])
        peaks, _ = scipy_signal.find_peaks(vals)
        sm10     = np.convolve(vals, np.ones(min(10, n)) / min(10, n), mode='same')
        sm20     = np.convolve(vals, np.ones(min(20, n)) / min(20, n), mode='same')
        p10, _   = scipy_signal.find_peaks(sm10)
        p20, _   = scipy_signal.find_peaks(sm20)
        dp,  _   = scipy_signal.find_peaks(np.abs(d))
        dp2, _   = scipy_signal.find_peaks(np.abs(d2))
        v        = float(np.var(vals))
        return [
            float(np.mean(vals)), v, float(np.std(vals)),
            float(kurtosis(vals)), float(scipy_skew(vals)),
            max(1, len(peaks)),  max(0, len(p10)), max(0, len(p20)),
            max(1, len(dp)),     max(0, len(dp2)),
            float(np.var(d)),    float(np.var(d2)),
            float(n * n),        float(n * sampling),
            v / duration if duration > 0 else 0.0,
            v / n        if n > 0        else 0.0,
        ]

    df = df.copy()
    if 'channel' not in df.columns:
        df['channel'] = 'CH_001'
    if 'segment' not in df.columns:
        df['segment'] = 0
    if 'value' not in df.columns:
        blacklist = {'segment', 'index', 'id', 'row', 'anomaly', 'train'}
        num_cols  = [c for c in df.select_dtypes(include='number').columns
                     if c.lower() not in blacklist]
        if not num_cols:
            raise ValueError("No numeric value column found in CSV")
        df['value'] = df[num_cols[0]]

    tensors, meta = [], []
    for (ch, sid), grp in df.groupby(['channel', 'segment'], sort=False):
        if 'timestamp' in grp.columns:
            grp = grp.sort_values('timestamp')
        vals     = grp['value'].values.astype(np.float32)
        n        = len(vals)
        sampling = int(grp['sampling'].iloc[0]) if 'sampling' in grp.columns else 5

        if n >= max_len:
            signal_padded = vals[:max_len].reshape(-1, 1)
        else:
            signal_padded = np.vstack([
                vals.reshape(-1, 1),
                np.zeros((max_len - n, 1), dtype=np.float32),
            ])

        stats       = np.array(_compute(vals, sampling), dtype=np.float32)
        stats_tiled = np.tile(stats, (max_len, 1))
        combined    = np.hstack([signal_padded, stats_tiled])
        tensors.append(combined)

        row = {'channel': ch, 'segment': int(sid)}
        row['anomaly'] = int(grp['anomaly'].iloc[0]) if 'anomaly' in grp.columns else -1
        row['train']   = int(grp['train'].iloc[0])   if 'train'   in grp.columns else 0
        meta.append(row)

    X       = np.array(tensors, dtype=np.float32)
    meta_df = pd.DataFrame(meta)

    N, L, F = X.shape
    flat    = X.reshape(-1, F)

    # ── Scaler guard ──────────────────────────────────────────────────────
    signal_scaler, stat_scaler, refitted = _get_compatible_scalers(
        flat[:, :1], flat[:, 1:], signal_scaler, stat_scaler
    )

    flat[:, :1] = signal_scaler.transform(flat[:, :1])
    flat[:, 1:] = stat_scaler.transform(flat[:, 1:])
    X = flat.reshape(N, L, F)

    return X, meta_df, refitted


def prepare_single_telemetry_data(df, signal_scaler, stat_scaler, max_len=100):
    """Fallback for CSVs without a segment column — treat as one big segment."""
    df = df.copy()
    df['channel'] = 'CH_001'
    df['segment'] = 0
    if 'value' not in df.columns:
        blacklist = {'segment', 'index', 'id', 'row', 'anomaly', 'train'}
        num_cols  = [c for c in df.select_dtypes(include='number').columns
                     if c.lower() not in blacklist]
        if not num_cols:
            raise ValueError("No numeric value column found in CSV")
        df['value'] = df[num_cols[0]]

    return prepare_segments_from_csv(df, signal_scaler, stat_scaler, max_len)