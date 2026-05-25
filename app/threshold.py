"""
threshold.py
============
Threshold selection and evaluation utilities.

Strategy
--------
After training on normal-only data, we compute reconstruction MSE for ALL
training segments (normal + anomaly). We then sweep 300 candidate thresholds
from the 50th to 99.9th percentile of those errors and pick the one that
maximises F1 on the training labels.

Why F1 and not accuracy?
  - Class imbalance: ~80% normal, ~20% anomaly in training set
  - F1 balances precision (not over-flagging) and recall (not missing faults)
  - A high-accuracy model that just predicts "normal" always would score ~80% —
    F1 correctly penalises that
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, fbeta_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)


def compute_reconstruction_errors(model,
                                  X: np.ndarray,
                                  batch_size: int = 64) -> np.ndarray:
    """
    Per-segment mean MSE across all timesteps and all features.

    Parameters
    ----------
    model      : trained Keras model
    X          : (N, max_len, n_features)
    batch_size : inference batch size

    Returns
    -------
    errors : (N,) float64
    """
    X_pred = model.predict(X, batch_size=batch_size, verbose=0)
    errors = np.mean(np.square(X - X_pred), axis=(1, 2))
    return errors.astype(np.float64)


def find_best_threshold(errors: np.ndarray,
                        labels: np.ndarray,
                        n_candidates: int = 300) -> dict:
    """
    Sweep thresholds and return the one that maximises F2 (favors recall).

    Parameters
    ----------
    errors       : reconstruction errors, shape (N,)
    labels       : ground-truth binary labels (1=anomaly), shape (N,)
    n_candidates : how many threshold values to try (300 is fine-grained enough)

    Returns
    -------
    dict with keys:
        threshold   : float — best threshold value
        percentile  : float — which percentile that corresponds to
        f1          : float — F1 score at best threshold
        f2          : float — F2 score at best threshold
        precision   : float
        recall      : float
        results_df  : DataFrame — full sweep results (for plotting)
    """
    percentiles = np.linspace(50, 99.9, n_candidates)
    thresholds  = np.percentile(errors, percentiles)

    best    = {'f2': -1.0}
    records = []

    for pct, thr in zip(percentiles, thresholds):
        preds = (errors > thr).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        f2    = fbeta_score(labels, preds, beta=2, zero_division=0)
        prec  = precision_score(labels, preds, zero_division=0)
        rec   = recall_score(labels, preds, zero_division=0)
        records.append({'percentile': pct, 'threshold': thr,
                        'f1': f1, 'f2': f2, 'precision': prec, 'recall': rec})
        if f2 > best['f2']:
            best = {
                'f1'        : f1,
                'f2'        : f2,
                'precision' : prec,
                'recall'    : rec,
                'threshold' : thr,
                'percentile': pct,
            }

    best['results_df'] = pd.DataFrame(records)
    return best


def evaluate(errors: np.ndarray,
             labels: np.ndarray,
             threshold: float,
             split_name: str = 'SET') -> dict:
    """
    Print a full evaluation report and return metrics as a dict.

    Parameters
    ----------
    errors     : reconstruction errors (N,)
    labels     : ground-truth binary labels (N,)
    threshold  : decision boundary
    split_name : label for the printed report header

    Returns
    -------
    metrics dict with tp, fp, tn, fn, precision, recall, f1, etc.
    """
    preds = (errors > threshold).astype(int)
    f1    = f1_score(labels, preds, zero_division=0)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    w = 62
    print(f'\n{"="*w}')
    print(f'  EVALUATION — {split_name}')
    print(f'{"="*w}')
    print(f'  Threshold              : {threshold:.8f}')
    print(f'  Total segments         : {len(labels):,}')
    print(f'  True  Positives  (TP)  : {tp:>6}   anomaly correctly flagged')
    print(f'  False Positives  (FP)  : {fp:>6}   normal wrongly flagged')
    print(f'  True  Negatives  (TN)  : {tn:>6}   normal correctly passed')
    print(f'  False Negatives  (FN)  : {fn:>6}   anomaly missed')
    print(f'  Precision              : {prec:.4f}')
    print(f'  Recall                 : {rec:.4f}')
    print(f'  F1 Score               : {f1:.4f}')
    print(f'\n  Mean recon error (normal)  : {errors[labels==0].mean():.8f}')
    print(f'  Mean recon error (anomaly) : {errors[labels==1].mean():.8f}')
    print(f'  Max  recon error           : {errors.max():.8f}')
    print(f'{"="*w}\n')

    return {
        'split'            : split_name,
        'threshold'        : float(threshold),
        'n_total'          : int(len(labels)),
        'tp'               : int(tp),
        'fp'               : int(fp),
        'tn'               : int(tn),
        'fn'               : int(fn),
        'precision'        : float(prec),
        'recall'           : float(rec),
        'f1'               : float(f1),
        'mean_err_normal'  : float(errors[labels == 0].mean()),
        'mean_err_anomaly' : float(errors[labels == 1].mean()),
        'max_err'          : float(errors.max()),
    }
