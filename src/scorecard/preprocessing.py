"""End-to-end preprocessing helpers for scoring."""

from __future__ import annotations

import pandas as pd

from scorecard.feature_engineering import engineer_features
from scorecard.woe import get_woe_columns, transform_woe


def prepare_model_frame(
    data: pd.DataFrame,
    binning_objects: dict,
    features: list[str] | None = None,
    cap_rules: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Engineer features, apply WOE, and return only model input columns."""
    engineered = engineer_features(data, cap_rules=cap_rules)
    with_woe = transform_woe(engineered, binning_objects=binning_objects, features=features)
    return with_woe[get_woe_columns(features)]
