"""Weight of Evidence transformation using fitted OptBinning objects."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scorecard.feature_engineering import FINAL_MODEL_FEATURES
from scorecard.utils import load_pickle, project_path


DEFAULT_WOE_FEATURES = FINAL_MODEL_FEATURES
DEFAULT_WOE_COLUMNS = [f"{col}_woe" for col in DEFAULT_WOE_FEATURES]


def load_woe_bins(path: str | Path | None = None) -> dict:
    """Load fitted OptBinning objects."""
    bins_path = Path(path) if path is not None else project_path("models", "woe_bins.pkl")
    return load_pickle(bins_path)


def transform_woe(
    data: pd.DataFrame,
    binning_objects: dict,
    features: list[str] | None = None,
    suffix: str = "_woe",
) -> pd.DataFrame:
    """Append WOE columns for selected raw or engineered features."""
    result = data.copy()
    selected_features = DEFAULT_WOE_FEATURES if features is None else features
    missing_columns = [col for col in selected_features if col not in result.columns]
    missing_bins = [col for col in selected_features if col not in binning_objects]

    if missing_columns:
        raise ValueError(f"Missing columns required for WOE transform: {missing_columns}")
    if missing_bins:
        raise ValueError(f"Missing fitted WOE binning objects: {missing_bins}")

    for col in selected_features:
        result[f"{col}{suffix}"] = binning_objects[col].transform(result[col], metric="woe")

    return result


def get_woe_columns(features: list[str] | None = None, suffix: str = "_woe") -> list[str]:
    selected_features = DEFAULT_WOE_FEATURES if features is None else features
    return [f"{col}{suffix}" for col in selected_features]
