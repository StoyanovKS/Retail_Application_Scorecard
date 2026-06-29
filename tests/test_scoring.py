import numpy as np

from scorecard.model import ScorecardModel
from scorecard.scoring import pd_to_score, score_to_pd


def test_pd_to_score_is_monotonic_decreasing():
    scores = pd_to_score(np.array([0.05, 0.10, 0.20, 0.40]))
    assert list(scores) == sorted(scores, reverse=True)


def test_score_to_pd_round_trip():
    pd_values = np.array([0.05, 0.20, 0.60])
    scores = pd_to_score(pd_values)
    recovered = score_to_pd(scores)
    assert np.allclose(pd_values, recovered)


def test_scorecard_model_loads_assets():
    model = ScorecardModel.from_files()
    assert model.features
    assert model.woe_columns == [f"{feature}_woe" for feature in model.features]
    assert hasattr(model.logistic_model, "predict_proba")
