"""
rule_engine.py
==============
Layer 1 Operational Rule Engine.
Checks for physical telemetry violations (spikes, flatlines, missing data).
"""
import numpy as np
from config import (
    SPIKE_ZSCORE_THRESHOLD, FLATLINE_MAX_STD, FLATLINE_MIN_WINDOW,
    RATE_OF_CHANGE_THRESHOLD
)
def check_window_rules(raw_window: np.ndarray) -> dict:
    """
    Apply rule-based detection to a single window of raw telemetry.
    
    Returns
    -------
    dict with boolean flags and descriptions.
    """
    flags = {
        "spike_flag": False,
        "flatline_flag": False,
        "range_flag": False,
        "details": []
    }
    
    if len(raw_window) == 0:
        return flags
        
    mean_val = np.mean(raw_window)
    std_val = np.std(raw_window)
    
    # 1. Spike / Outlier detection (Z-Score)
    if std_val > 1e-6:
        z_scores = np.abs((raw_window - mean_val) / std_val)
        if np.any(z_scores > SPIKE_ZSCORE_THRESHOLD):
            flags["spike_flag"] = True
            flags["details"].append(f"Spike detected (Z > {SPIKE_ZSCORE_THRESHOLD})")
            
    # 2. Flatline / Stuck sensor detection
    if std_val < FLATLINE_MAX_STD:
        flags["flatline_flag"] = True
        flags["details"].append("Flatline / stuck sensor detected")
        
    # 3. Sudden rate of change
    diffs = np.abs(np.diff(raw_window))
    if mean_val != 0 and len(diffs) > 0:
        pct_change = np.max(diffs) / np.abs(mean_val)
        if pct_change > RATE_OF_CHANGE_THRESHOLD:
            flags["range_flag"] = True
            flags["details"].append(f"Sudden change (> {RATE_OF_CHANGE_THRESHOLD*100}%)")
            
    return flags