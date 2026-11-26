"""
Stage 4 monitoring: drift (PSI) and per-group error.

PSI (Population Stability Index), 10 bins:
  * bin edges = deciles of the REFERENCE data (cleaned train.csv); the outer
    edges are opened to -inf / +inf so every new value lands in some bin
  * expected % = share of train rows in each bin, actual % = share of later rows
  * PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)
  * a tiny epsilon stops ln(0) when a bin is empty
Rule of thumb: < 0.1 stable, 0.1 - 0.25 moderate shift, > 0.25 big shift.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

EPS = 1e-4


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected = np.clip(expected, EPS, None)
    actual = np.clip(actual, EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_table(reference: pd.DataFrame, current: pd.DataFrame,
              columns: list[str], bins: int = 10) -> dict:
    return {c: round(psi(reference[c], current[c], bins), 4) for c in columns}


def mae_by_group(model, df: pd.DataFrame, features: list[str], target: str,
                 group_col: str) -> dict:
    preds = model.predict(df[features])
    out = {}
    for group, idx in df.groupby(group_col).groups.items():
        rows = df.index.get_indexer(idx)
        out[group] = round(float(mean_absolute_error(df[target].iloc[rows], preds[rows])), 2)
    return out
