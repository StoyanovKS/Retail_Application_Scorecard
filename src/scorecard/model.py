"""Model wrapper combining feature engineering, WOE and score scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scorecard.feature_engineering import DEFAULT_CAP_RULES, FINAL_MODEL_FEATURES
from scorecard.preprocessing import prepare_model_frame
from scorecard.scoring import BASE_ODDS, BASE_SCORE, PDO, pd_to_score
from scorecard.utils import load_config, load_pickle, project_path
from scorecard.woe import get_woe_columns


@dataclass
class ScorecardModel:
    """Production wrapper for the trained credit scorecard."""

    logistic_model: object
    woe_bins: dict
    features: list[str]
    base_score: float = BASE_SCORE
    base_odds: float = BASE_ODDS
    pdo: float = PDO
    cap_rules: dict[str, dict[str, float]] | None = None

    @classmethod
    def from_files(
        cls,
        model_path: str | Path | None = None,
        woe_bins_path: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> "ScorecardModel":
        """Load the scorecard from model assets and config."""
        config = load_config(config_path)
        model_cfg = config.get("model", {})
        scoring_cfg = config.get("score_scaling", {})
        feature_cfg = config.get("features", {})

        resolved_model_path = _resolve_asset_path(
            model_path or model_cfg.get("logistic_model_path", project_path("models", "logistic_model.pkl"))
        )
        resolved_bins_path = _resolve_asset_path(
            woe_bins_path or model_cfg.get("woe_bins_path", project_path("models", "woe_bins.pkl"))
        )

        return cls(
            logistic_model=load_pickle(resolved_model_path),
            woe_bins=load_pickle(resolved_bins_path),
            features=list(feature_cfg.get("final_model_features", FINAL_MODEL_FEATURES)),
            base_score=float(scoring_cfg.get("base_score", BASE_SCORE)),
            base_odds=float(scoring_cfg.get("base_odds", BASE_ODDS)),
            pdo=float(scoring_cfg.get("pdo", PDO)),
            cap_rules=feature_cfg.get("cap_rules", DEFAULT_CAP_RULES),
        )

    @property
    def woe_columns(self) -> list[str]:
        return get_woe_columns(self.features)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return model-ready WOE features."""
        return prepare_model_frame(
            data,
            binning_objects=self.woe_bins,
            features=self.features,
            cap_rules=self.cap_rules,
        )

    def predict_pd(self, data: pd.DataFrame) -> pd.Series:
        """Predict probability of default for application records."""
        model_frame = self.transform(data)
        pd_values = self.logistic_model.predict_proba(model_frame)[:, 1]
        return pd.Series(pd_values, index=data.index, name="pd_pred")

    def predict_score(self, data: pd.DataFrame) -> pd.Series:
        """Predict scorecard points for application records."""
        pd_values = self.predict_pd(data)
        scores = pd_to_score(
            pd_values,
            base_score=self.base_score,
            base_odds=self.base_odds,
            pdo=self.pdo,
        )
        scores.name = "score"
        return scores

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return PD and score predictions."""
        pd_values = self.predict_pd(data)
        scores = pd_to_score(
            pd_values,
            base_score=self.base_score,
            base_odds=self.base_odds,
            pdo=self.pdo,
        )
        return pd.DataFrame({"pd_pred": pd_values, "score": scores}, index=data.index)


def _resolve_asset_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return project_path(*resolved.parts)
