"""
train.py
========
Train the LSTM Autoencoder on normal segments from segments.csv + dataset.csv.

What it does
------------
1.  Loads both CSVs via data_loader.py
2.  Builds tensors:
      X_train      — normal-only training segments
      X_val        — 10% held-out normal segments (early stopping)
      X_all_train  — all train segments (normal + anomaly) for threshold tuning
3.  Trains LSTM Autoencoder (model.py)
4.  Sweeps 300 threshold candidates, picks F1-maximising one (threshold.py)
5.  Evaluates on full training set, prints confusion matrix
6.  Saves everything to saved_model/:
      lstm_autoencoder.keras
      signal_scaler.pkl
      stat_scaler.pkl
      metadata.json
      loss_curve.png
      train_error_plot.png
      threshold_sweep.png

Usage
-----
    python train.py \\
        --segments  data/segments.csv \\
        --dataset   data/dataset.csv  \\
        --max_len   100               \\
        --epochs    50                \\
        --batch     32                \\
        --latent    64                \\
        --out_dir   saved_model
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import prepare_training_data
from model       import build_lstm_autoencoder, get_callbacks
from threshold   import compute_reconstruction_errors, find_best_threshold, evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(history, out_dir: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history['loss'],     label='Train loss', color='#4C9BE8', linewidth=1.5)
    ax.plot(history.history['val_loss'], label='Val loss',   color='#E8A84C', linewidth=1.5)
    ax.set_title('Training & Validation Loss (MSE)', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'loss_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[PLOT] {path}')


def plot_train_errors(errors: np.ndarray,
                      labels: np.ndarray,
                      threshold: float,
                      out_dir: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    # ── Scatter: error per segment ──────────────────────────────────────
    normal_idx  = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    ax1.scatter(normal_idx,  errors[normal_idx],
                s=10, color='#4C9BE8', label='Normal',  alpha=0.6, zorder=2)
    ax1.scatter(anomaly_idx, errors[anomaly_idx],
                s=10, color='#E84C4C', label='Anomaly', alpha=0.8, zorder=3)
    ax1.axhline(threshold, color='#2CA02C', linewidth=1.5, linestyle='--',
                label=f'Threshold = {threshold:.6f}')
    ax1.set_title('Reconstruction Error — All Training Segments', fontsize=13)
    ax1.set_xlabel('Segment index')
    ax1.set_ylabel('MSE')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Histogram: error distribution by class ──────────────────────────
    ax2.hist(errors[labels == 0], bins=60, color='#4C9BE8',
             alpha=0.7, label='Normal',  edgecolor='white', linewidth=0.3)
    ax2.hist(errors[labels == 1], bins=60, color='#E84C4C',
             alpha=0.7, label='Anomaly', edgecolor='white', linewidth=0.3)
    ax2.axvline(threshold, color='#2CA02C', linewidth=1.5, linestyle='--',
                label='Threshold')
    ax2.set_title('Error Distribution by Class', fontsize=13)
    ax2.set_xlabel('MSE')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, 'train_error_plot.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[PLOT] {path}')


def plot_threshold_sweep(sweep_df: pd.DataFrame,
                         best_threshold: float,
                         out_dir: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(sweep_df['threshold'], sweep_df['f1'],
            color='#4C9BE8', label='F1',        linewidth=1.5)
    ax.plot(sweep_df['threshold'], sweep_df['precision'],
            color='#E8A84C', label='Precision', linewidth=1.2, linestyle='--')
    ax.plot(sweep_df['threshold'], sweep_df['recall'],
            color='#E84C4C', label='Recall',    linewidth=1.2, linestyle=':')
    ax.axvline(best_threshold, color='#2CA02C', linewidth=1.5, linestyle='--',
               label=f'Best threshold = {best_threshold:.6f}')
    ax.set_title('Threshold Sweep — F1 / Precision / Recall', fontsize=13)
    ax.set_xlabel('Threshold (MSE)')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, 'threshold_sweep.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'[PLOT] {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM Autoencoder on telemetry segments')
    parser.add_argument('--segments', default='segments.csv',
                        help='Path to segments.csv')
    parser.add_argument('--dataset',  default='dataset.csv',
                        help='Path to dataset.csv')
    parser.add_argument('--max_len',  type=int, default=75,
                        help='Pad/truncate segments to this many timesteps (default 75)')
    parser.add_argument('--epochs',   type=int, default=50,
                        help='Max training epochs (EarlyStopping may stop sooner)')
    parser.add_argument('--batch',    type=int, default=32,
                        help='Batch size')
    parser.add_argument('--latent',   type=int, default=64,
                        help='LSTM bottleneck dimension')
    parser.add_argument('--out_dir',  default='saved_model',
                        help='Directory to save model, scalers, and plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Step 1: Load and preprocess ──────────────────────────────────────
    print('\n' + '='*60)
    print('  STEP 1 / 5 — Loading and preprocessing data')
    print('='*60)
    (X_train, X_val,
     X_all_train, meta_all_train,
     signal_scaler, stat_scaler) = prepare_training_data(
        segments_path=args.segments,
        dataset_path=args.dataset,
        max_len=args.max_len,
        save_dir=args.out_dir,
    )
    n_features = X_train.shape[2]
    print(f'\n  Input shape : (N, {args.max_len}, {n_features})')
    print(f'  X_train     : {X_train.shape}  ← normal-only, LSTM trains on this')
    print(f'  X_val       : {X_val.shape}    ← held-out normal (early stopping)')
    print(f'  X_all_train : {X_all_train.shape}  ← normal + anomaly (threshold tuning)')

    # ── Step 2: Build model ──────────────────────────────────────────────
    print('\n' + '='*60)
    print('  STEP 2 / 5 — Building model')
    print('='*60)
    model = build_lstm_autoencoder(
        max_len=args.max_len,
        n_features=n_features,
        latent_dim=args.latent,
    )
    model.summary()

    # ── Step 3: Train (Pass 1) ───────────────────────────────────────────
    print('\n' + '='*60)
    print(f'  STEP 3 / 6 — Pass 1 Training (10 epochs max) to find noisy normal data')
    print('='*60)
    # Train briefly just to identify the most confusing "normal" segments
    history1 = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=min(10, args.epochs),
        batch_size=args.batch,
        callbacks=get_callbacks(patience_stop=3, patience_lr=2),
        shuffle=True,
        verbose=1,
    )
    
    # Predict on all normal data and find the highest MSEs
    print('\n  Identifying noisy normal segments...')
    from threshold import compute_reconstruction_errors
    normal_errors = compute_reconstruction_errors(model, X_train)
    
    # 98th percentile -> keep 98%, discard top 2%
    cutoff = np.percentile(normal_errors, 98)
    keep_idx = np.where(normal_errors <= cutoff)[0]
    discard_count = len(normal_errors) - len(keep_idx)
    
    print(f'  Discarding {discard_count} segments with MSE > {cutoff:.6f}')
    
    X_train_clean = X_train[keep_idx]
    
    # Rebuild model from scratch
    print('\n' + '='*60)
    print('  STEP 4 / 6 — Rebuilding model for Pass 2')
    print('='*60)
    model = build_lstm_autoencoder(
        max_len=args.max_len,
        n_features=n_features,
        latent_dim=args.latent,
    )
    
    print('\n' + '='*60)
    print(f'  STEP 5 / 6 — Pass 2 Training on clean data (up to {args.epochs} epochs)')
    print('='*60)
    history = model.fit(
        X_train_clean, X_train_clean,
        validation_data=(X_val, X_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=get_callbacks(),
        shuffle=True,
        verbose=1,
    )
    epochs_run = len(history.history['loss'])
    print(f'\n  Stopped at epoch : {epochs_run}')
    print(f'  Best val loss    : {min(history.history["val_loss"]):.8f}')
    plot_loss_curve(history, args.out_dir)

    # ── Step 6: Threshold selection ──────────────────────────────────────
    print('\n' + '='*60)
    print('  STEP 6 / 6 — Computing errors + tuning threshold')
    print('='*60)
    train_errors = compute_reconstruction_errors(model, X_all_train)
    train_labels = meta_all_train['anomaly'].values

    mean_normal  = train_errors[train_labels == 0].mean()
    mean_anomaly = train_errors[train_labels == 1].mean()
    sep_ratio    = mean_anomaly / mean_normal if mean_normal > 0 else float('nan')
    print(f'\n  Mean MSE (normal)  : {mean_normal:.8f}')
    print(f'  Mean MSE (anomaly) : {mean_anomaly:.8f}')
    print(f'  Separation ratio   : {sep_ratio:.2f}×  (higher is better)')

    best      = find_best_threshold(train_errors, train_labels, n_candidates=300)
    threshold = best['threshold']
    print(f'\n  Best threshold   : {threshold:.8f}  ({best["percentile"]:.1f}th pct)')
    print(f'  Train F1         : {best["f1"]:.4f}')
    print(f'  Train F2         : {best["f2"]:.4f}')
    print(f'  Train Precision  : {best["precision"]:.4f}')
    print(f'  Train Recall     : {best["recall"]:.4f}')

    plot_train_errors(train_errors, train_labels, threshold, args.out_dir)
    plot_threshold_sweep(best['results_df'], threshold, args.out_dir)

    train_metrics = evaluate(train_errors, train_labels, threshold,
                             split_name='TRAINING SET')

    # ── Step 7: Save ─────────────────────────────────────────────────────
    print('='*60)
    print('  STEP 7 / 7 — Saving artifacts')
    print('='*60)

    model_path = os.path.join(args.out_dir, 'lstm_autoencoder.keras')
    model.save(model_path)
    print(f'  Model    → {model_path}')

    metadata = {
        'max_len'          : args.max_len,
        'n_features'       : n_features,
        'latent_dim'       : args.latent,
        'threshold'        : float(threshold),
        'threshold_pct'    : float(best['percentile']),
        'epochs_trained'   : epochs_run,
        'final_train_loss' : float(history.history['loss'][-1]),
        'final_val_loss'   : float(history.history['val_loss'][-1]),
        'best_val_loss'    : float(min(history.history['val_loss'])),
        'sep_ratio'        : float(sep_ratio),
        'train_metrics'    : train_metrics,
        'stat_features'    : [
            'mean', 'var', 'std', 'kurtosis', 'skew',
            'n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks',
            'diff_peaks', 'diff2_peaks',
            'diff_var', 'diff2_var',
            'gaps_squared', 'len_weighted',
            'var_div_duration', 'var_div_len',
        ],
    }
    meta_path = os.path.join(args.out_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'  Metadata → {meta_path}')

    print('\n' + '='*60)
    print('  TRAINING COMPLETE')
    print('='*60)
    print(f'\n  Saved to: {args.out_dir}/')
    print(f'    lstm_autoencoder.keras')
    print(f'    signal_scaler.pkl + stat_scaler.pkl')
    print(f'    metadata.json  (threshold={threshold:.6f}, F1={best["f1"]:.4f})')
    print(f'    loss_curve.png | train_error_plot.png | threshold_sweep.png')
    print(f'\n  Next step:')
    print(f'    python test.py --segments <test_segments.csv> --dataset <test_dataset.csv>\n')


if __name__ == '__main__':
    main()
