import pandas as pd

from scorecard.feature_engineering import engineer_features
from scorecard.woe import load_woe_bins, transform_woe


def test_transform_woe_creates_final_model_columns():
    raw = pd.DataFrame(
        {
            "issue_d": ["2017-01-01", "2018-02-01"],
            "earliest_cr_line": ["2008-01-01", "2012-03-01"],
            "term": ["36 months", "60 months"],
            "loan_amnt": [10000.0, 24000.0],
            "annual_inc": [70000.0, 45000.0],
            "dti": [14.0, 31.0],
            "revol_util": [35.0, 82.0],
            "inq_last_6mths": [0.0, 3.0],
            "acc_open_past_24mths": [2.0, 8.0],
            "mort_acc": [2.0, 0.0],
            "purpose": ["debt_consolidation", "credit_card"],
            "home_ownership": ["MORTGAGE", "RENT"],
        }
    )

    bins = load_woe_bins()
    engineered = engineer_features(raw)
    transformed = transform_woe(engineered, bins)

    expected = {
        "loan_to_income_woe",
        "dti_woe",
        "revol_util_woe",
        "annual_inc_woe",
        "inq_last_6mths_woe",
        "acc_open_past_24mths_woe",
        "mort_acc_woe",
        "purpose_woe",
        "home_ownership_woe",
        "term_woe",
    }
    assert expected.issubset(transformed.columns)
    assert transformed[list(expected)].notna().all().all()
