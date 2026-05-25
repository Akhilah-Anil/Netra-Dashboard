# LSTM Autoencoder — Telemetry Anomaly Detection

Train on labelled normal segments. Test on any CSV. Get F1 / precision / recall.

---

## Project structure

```
lstm_project/
├── train.py          ← run this first
├── test.py           ← run this after training
├── data_loader.py    ← data pipeline (no edits needed)
├── model.py          ← LSTM autoencoder architecture
├── threshold.py      ← F1-maximising threshold + evaluation
├── requirements.txt
└── README.md
```

---

## What the data looks like (confirmed from EDA)

| File | Rows | Key columns |
|------|------|-------------|
| `segments.csv` | 303,493 | channel, timestamp, value, anomaly, segment, train |
| `dataset.csv` | 2,123 | segment, channel, anomaly, train, mean, var, std, … (16 stat features) |

- **2,123 segments** across 9 channels
- **train split**: 1,594 train / 529 test (via `train` column, already baked in)
- **Normal train segments**: 1,273 | **Anomaly train**: 321
- Segment lengths: min=8, max=1040, median≈70, p95=464
- Every segment is fully homogeneous — all rows share the same anomaly label

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Train

```bash
python train.py \
    --segments  segments.csv \
    --dataset   dataset.csv  \
    --max_len   100          \
    --epochs    50           \
    --batch     32           \
    --latent    64           \
    --out_dir   saved_model
```

What gets saved to `saved_model/`:

```
lstm_autoencoder.keras   ← model weights
signal_scaler.pkl        ← MinMaxScaler fitted on normal training signal
stat_scaler.pkl          ← MinMaxScaler fitted on normal training stats
metadata.json            ← threshold, F1, all config
loss_curve.png
train_error_plot.png     ← normal vs anomaly error scatter + histogram
threshold_sweep.png      ← F1/precision/recall across 300 threshold candidates
```

### 3. Test

```bash
# Uses train==0 rows from the same files:
python test.py \
    --segments  segments.csv \
    --dataset   dataset.csv  \
    --model_dir saved_model  \
    --out_dir   results

# Or point to completely new files:
python test.py \
    --segments  new_segments.csv \
    --dataset   new_dataset.csv  \
    --model_dir saved_model
```

If the test CSV has an `anomaly` column → full F1 / precision / recall.
If not → inference-only mode (flags segments, no ground-truth comparison).

What gets saved to `results/`:

```
all_segments.csv         ← every segment: error, predicted_anomaly, deviation
anomalies.csv            ← only flagged segments
test_error_plot.png
test_metrics.json        ← (only if anomaly labels present)
```

---

## How it works

```
segments.csv + dataset.csv
         │
         ▼
   data_loader.py
   ─────────────────────────────────────
   Filter: train==1 AND anomaly==0       → 1,273 normal segments
   For each segment:
     raw value time-series (T, 1)
     + 16 stats from dataset.csv (T, 16)  ← replicated across timesteps
     → pad/truncate to max_len
     → shape (max_len, 17)
   MinMax scale signal and stats separately
   → X_train (N_train, 100, 17)
   → X_val   (N_val,   100, 17)
   → X_all_train (normal + anomaly, for threshold tuning)
         │
         ▼
   model.py — LSTM Autoencoder
   ─────────────────────────────────────
   Input  (batch, 100, 17)
   LSTM(128, seq=True) → Dropout(0.2)
   LSTM(64,  seq=False)  ← bottleneck
   RepeatVector(100)
   LSTM(64,  seq=True)
   Dropout(0.2)
   LSTM(128, seq=True)
   TimeDistributed(Dense(17))
   Output (batch, 100, 17)
   Loss: MSE — trained on normal only
         │
         ▼
   threshold.py
   ─────────────────────────────────────
   Compute MSE for all train segments (normal + anomaly)
   Sweep 300 percentile candidates (50th → 99.9th)
   Pick threshold that maximises F1 on training labels
   Save threshold to metadata.json
         │
         ▼
   test.py — no retraining, no refitting
   ─────────────────────────────────────
   Load saved model + scalers + threshold
   Transform new segments with saved scalers
   Predict → compute MSE per segment
   Flag segments above threshold
   If labels present: print F1/precision/recall/confusion matrix
   Save all_segments.csv + anomalies.csv
```

---

## Key design choices

**Why train on normal-only?**
The autoencoder learns to compress and reconstruct normal patterns. When it sees
an anomalous segment, it can't reconstruct it well → high MSE → flagged.
Training on anomalies too would teach the model to reconstruct them, destroying
the signal.

**Why F1-maximising threshold (not fixed percentile)?**
The dataset is imbalanced (~80% normal, ~20% anomaly). A fixed 95th percentile
might miss most anomalies or flag too many normals. Sweeping and maximising F1
on the training labels (where ground truth is known) gives an optimal threshold
for this specific dataset.

**Why 16 stat features alongside the raw signal?**
The raw signal alone can look similar across different anomaly types. The stat
features (variance, kurtosis, peak counts, etc.) give the encoder context about
the *shape* of the segment, helping it form a tighter "normal" manifold in latent
space and produce a better-separated error distribution.

**Two-file design (segments + dataset)?**
`segments.csv` has the time-series rows. `dataset.csv` has pre-computed features
per segment. Keeping them separate is efficient — the LSTM only needs the signal
for temporal modelling; the stat features don't need to be recomputed per timestep.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Key mismatch` error | Check that (channel, segment) pairs are identical in both CSVs |
| Very low F1 (<0.3) | Try increasing `--epochs` or `--max_len`; check stat features for NaN |
| All segments flagged | Run with higher `--threshold` override or check for data drift |
| No anomalies flagged | Lower the threshold: `test.py --threshold 0.00001` |
| OOM on GPU | Reduce `--batch` to 16 |
