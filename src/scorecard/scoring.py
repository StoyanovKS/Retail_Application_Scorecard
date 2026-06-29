"""Score scaling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_SCORE = 600.0
BASE_ODDS = 20.0
PDO = 50.0
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def pd_to_score(
    pd_values,
    base_score: float = BASE_SCORE,
    base_odds: float = BASE_ODDS,
    pdo: float = PDO,
):
    """Convert probability of default to scorecard points."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    clipped = np.clip(pd_values, 1e-6, 1 - 1e-6)
    odds = (1 - clipped) / clipped
    scores = offset + factor * np.log(odds)
    return _preserve_pandas_type(pd_values, scores)


def score_to_pd(
    scores,
    base_score: float = BASE_SCORE,
    base_odds: float = BASE_ODDS,
    pdo: float = PDO,
):
    """Convert scorecard points back to probability of default."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    odds = np.exp((np.asarray(scores) - offset) / factor)
    pd_values = 1 / (1 + odds)
    return _preserve_pandas_type(scores, pd_values)


def _preserve_pandas_type(original, values):
    if isinstance(original, pd.Series):
        return pd.Series(values, index=original.index, name=original.name)
    return values
