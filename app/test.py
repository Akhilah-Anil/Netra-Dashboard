"""
test.py
=======
Inference script — loads the saved model and runs on new CSV inputs.
No retraining. No refitting of scalers. Threshold comes from training.

Usage
-----
    # Test set from same files (uses train==0 rows):
    python test.py \\
        --segments  segments.csv \\
        --dataset   dataset.csv  \\
        --model_dir saved_model

    # Completely new CSV (with anomaly labels — gets full F1 report):
    python test.py \\
        --segments  new_segments.csv \\
        --dataset   new_dataset.csv  \\
        --model_dir saved_model

    # Override threshold:
    python test.py --segments ... --dataset ... --threshold 0.00042

    # Custom output dir:
    python test.py --segments ... --dataset ... --out_dir my_results/

Input CSVs
----------
segments CSV must have columns:
    channel, timestamp, value, anomaly (optional), segment, train (optional)

dataset CSV must have columns:
    segment, channel, train (optional), anomaly (optional),
    mean, var, std, kurtosis, skew, n_peaks, smooth10_n_peaks,
    smooth20_n_peaks, diff_peaks, diff2_peaks, diff_var, diff2_var,
    gaps_squared, len_weighted, var_div_duration, var_div_len

If `anomaly` column is present and has both 0 and 1 values:
    → full evaluation (F1, precision, recall, confusion matrix)
Otherwise:
    → inference-only mode (flags segments, no ground-truth comparison)

Outputs (results/ by default)
------------------------------
    all_segments.csv    — every segment with its error, flag, deviation
    anomalies.csv       — only flagged segments
    test_error_plot.png — scatter + histogram plots
    test_metrics.json   — F1/precision/recall (if labels available)
"""

import os
import json
import argparse
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import prepare_test_data
from threshold   import compute_reconstruction_errors, evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_errors(errors: np.ndarray,
                     labels: np.ndarray,
                     threshold: float,
                     out_dir: str,
                     has_labels: bool):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    # ── Scatter ──────────────────────────────────────────────────────────
    if has_labels:
        normal_idx  = np.where(labels == 0)[0]
        anomaly_idx = np.where(labels == 1)[0]
        ax1.scatter(normal_idx,  errors[normal_idx],
                    s=12, color='#4C9BE8', label='Normal',  alpha=0.6, zorder=2)
        ax1.scatter(anomaly_idx, errors[anomaly_idx],
                    s=12, color='#E84C4C', label='Anomaly', alpha=0.8, zorder=3)
    else:
        flagged   = errors > threshold
        ax1.scatter(np.where(~flagged)[0], errors[~flagged],
                    s=12, color='#4C9BE8', label='Normal', alpha=0.6)
        ax1.scatter(np.where(flagged)[0],  errors[flagged],
                    s=12, color='#E84C4C', label='Flagged', alpha=0.8)

    ax1.axhline(threshold, color='#2CA02C', linewidth=1.5, linestyle='--',
                label=f'Threshold = {threshold:.6f}')
    ax1.set_title('Reconstruction Error — Test Segments', fontsize=13)
    ax1.set_xlabel('Segment index')
    ax1.set_ylabel('MSE')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Histogram ────────────────────────────────────────────────────────
    ax2.hist(errors, bins=60, color='#4C9BE8',
             edgecolor='white', linewidth=0.3, label='All segments')
    ax2.axvline(threshold, color='#2CA02C', linewidth=1.5, linestyle='--',
                label='Threshold')
    ax2.set_title('Error Distribution — Test Data', fontsize=13)
    ax2.set_xlabel('MSE')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'test_error_plot.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[PLOT] {path}')


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_results(errors: np.ndarray,
                 labels: np.ndarray,
                 threshold: float,
                 meta_df: pd.DataFrame,
                 out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    result_df = meta_df.copy()
    result_df['reconstruction_error']      = errors.round(8)
    result_df['predicted_anomaly']         = (errors > threshold).astype(int)
    result_df['deviation_above_threshold'] = np.maximum(0, errors - threshold).round(8)

    all_path = os.path.join(out_dir, 'all_segments.csv')
    result_df.to_csv(all_path, index=False)
    print(f'[SAVE] All segments    → {all_path}  ({len(result_df)} rows)')

    anom_df  = result_df[result_df['predicted_anomaly'] == 1].copy()
    anom_path = os.path.join(out_dir, 'anomalies.csv')
    anom_df.to_csv(anom_path, index=False)
    print(f'[SAVE] Flagged anomalies → {anom_path}  ({len(anom_df)} rows)')

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LSTM Autoencoder inference on new telemetry CSVs')
    parser.add_argument('--segments',   required=True,
                        help='Path to test segments.csv')
    parser.add_argument('--dataset',    required=True,
                        help='Path to test dataset.csv')
    parser.add_argument('--model_dir',  default='saved_model',
                        help='Directory containing saved model and scalers')
    parser.add_argument('--out_dir',    default='results',
                        help='Output directory for results and plots')
    parser.add_argument('--threshold',  type=float, default=None,
                        help='Override threshold (default: use value from training metadata)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: Load saved artifacts ─────────────────────────────────────
    print('\n' + '='*60)
    print('  STEP 1 / 4 — Loading saved model and artifacts')
    print('='*60)
    paths = {
        'model'         : os.path.join(args.model_dir, 'lstm_autoencoder.keras'),
        'metadata'      : os.path.join(args.model_dir, 'metadata.json'),
        'signal_scaler' : os.path.join(args.model_dir, 'signal_scaler.pkl'),
        'stat_scaler'   : os.path.join(args.model_dir, 'stat_scaler.pkl'),
    }
    for label, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'Missing artifact: {path}\n'
                f'Run train.py first to generate saved_model/')

    model = tf.keras.models.load_model(paths['model'])
    print(f'  Model loaded from   : {paths["model"]}')

    with open(paths['metadata']) as f:
        metadata = json.load(f)
    with open(paths['signal_scaler'], 'rb') as f:
        signal_scaler = pickle.load(f)
    with open(paths['stat_scaler'], 'rb') as f:
        stat_scaler = pickle.load(f)

    max_len   = metadata['max_len']
    threshold = args.threshold if args.threshold is not None else metadata['threshold']
    source    = 'overridden' if args.threshold is not None else 'from training'

    print(f'  max_len           : {max_len}')
    print(f'  threshold         : {threshold:.8f}  ({source})')
    print(f'  Training F1       : {metadata["train_metrics"]["f1"]:.4f}')
    print(f'  Training sep ratio: {metadata.get("sep_ratio", "n/a")}×')

    # ── Step 2: Load test data ────────────────────────────────────────────
    print('\n' + '='*60)
    print('  STEP 2 / 4 — Loading test data')
    print('='*60)
    X_test, meta_test = prepare_test_data(
        segments_path=args.segments,
        dataset_path=args.dataset,
        signal_scaler=signal_scaler,
        stat_scaler=stat_scaler,
        max_len=max_len,
    )

    # Ground-truth labels available?
    has_labels = (
        'anomaly' in meta_test.columns
        and meta_test['anomaly'].nunique() > 1
        and -1 not in meta_test['anomaly'].unique()
    )
    labels = meta_test['anomaly'].values if has_labels else np.zeros(len(meta_test), dtype=int)

    if has_labels:
        print(f'\n  Ground-truth labels detected — full F1 evaluation will run.')
        print(f'  Anomaly segments : {(labels==1).sum()} / {len(labels)}')
    else:
        print(f'\n  No ground-truth labels — inference-only mode.')

    # ── Step 3: Inference ─────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  STEP 3 / 4 — Running inference')
    print('='*60)
    errors    = compute_reconstruction_errors(model, X_test)
    n_flagged = int((errors > threshold).sum())

    print(f'\n  Total test segments  : {len(errors)}')
    print(f'  Mean reconstruction error : {errors.mean():.8f}')
    print(f'  Max  reconstruction error : {errors.max():.8f}')
    print(f'  Segments flagged as anomaly: {n_flagged} / {len(errors)} '
          f'({100*n_flagged/len(errors):.1f}%)')

    # ── Step 4: Evaluate + Save ───────────────────────────────────────────
    print('\n' + '='*60)
    print('  STEP 4 / 4 — Saving results')
    print('='*60)

    if has_labels:
        test_metrics = evaluate(errors, labels, threshold, split_name='TEST SET')
        metrics_path = os.path.join(args.out_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f'[SAVE] Metrics → {metrics_path}')
    else:
        print('  Skipping F1 evaluation (no ground-truth labels in this file).')

    plot_test_errors(errors, labels, threshold, args.out_dir, has_labels)
    save_results(errors, labels, threshold, meta_test, args.out_dir)

    print('\n' + '='*60)
    print('  INFERENCE COMPLETE')
    print('='*60)
    print(f'\n  Results saved to: {args.out_dir}/')
    print(f'    all_segments.csv  — every segment with error + flag')
    print(f'    anomalies.csv     — only flagged segments')
    print(f'    test_error_plot.png')
    if has_labels:
        print(f'    test_metrics.json')
    print()


if __name__ == '__main__':
    main()
