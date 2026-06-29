import numpy as np
import pandas as pd

from scorecard.feature_engineering import engineer_features, parse_term_months


def test_parse_term_months():
    assert parse_term_months("36 months") == 36
    assert parse_term_months(" 60 months") == 60
    assert np.isnan(parse_term_months(None))


def test_engineer_features_creates_training_consistent_fields():
    data = pd.DataFrame(
        {
            "issue_d": ["2017-01-01"],
            "earliest_cr_line": ["2010-01-01"],
            "term": ["60 months"],
            "loan_amnt": [12000.0],
            "annual_inc": [60000.0],
            "fico_range_high": [704.0],
            "revol_util": [85.0],
            "dti": [36.0],
            "delinq_2yrs": [1.0],
            "inq_last_6mths": [2.0],
            "pub_rec": [0.0],
            "pub_rec_bankruptcies": [0.0],
            "mort_acc": [1.0],
            "emp_length": ["10+ years"],
        }
    )

    result = engineer_features(data)

    assert result.loc[0, "term_months"] == 60
    assert result.loc[0, "is_60m_term"] == 1
    assert result.loc[0, "loan_to_income"] == 0.2
    assert result.loc[0, "fico_mid"] == 699
    assert result.loc[0, "credit_history_months"] == 84
    assert result.loc[0, "high_revol_util_flag"] == 1
    assert result.loc[0, "high_dti_flag"] == 1
    assert result.loc[0, "emp_length_num"] == 10
    assert "loan_to_income_capped" in result.columns
