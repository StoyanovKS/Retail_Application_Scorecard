"""Training-consistent feature engineering for scorecard inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_CAP_RULES: dict[str, dict[str, float]] = {
    "annual_inc": {"lower_p01": 22000.0, "upper_p99": 256000.0},
    "dti": {"lower_p01": 2.5, "upper_p99": 37.92},
    "revol_util": {"lower_p01": 2.5, "upper_p99": 98.0},
    "revol_bal": {"lower_p01": 407.0, "upper_p99": 89594.0},
    "loan_amnt": {"lower_p01": 1900.0, "upper_p99": 35000.0},
    "loan_to_income": {"lower_p01": 0.027027027, "upper_p99": 0.479591837},
    "credit_history_months": {"lower_p01": 48.0, "upper_p99": 455.0},
    "credit_history_years": {"lower_p01": 4.0, "upper_p99": 37.916666667},
    "open_acc": {"lower_p01": 4.0, "upper_p99": 30.0},
    "total_acc": {"lower_p01": 7.0, "upper_p99": 62.0},
    "inq_last_6mths": {"lower_p01": 0.0, "upper_p99": 4.0},
    "delinq_2yrs": {"lower_p01": 0.0, "upper_p99": 4.0},
    "pub_rec": {"lower_p01": 0.0, "upper_p99": 3.0},
    "mort_acc": {"lower_p01": 0.0, "upper_p99": 8.0},
}


FINAL_MODEL_FEATURES = [
    "loan_to_income",
    "dti",
    "revol_util",
    "annual_inc",
    "inq_last_6mths",
    "acc_open_past_24mths",
    "mort_acc",
    "purpose",
    "home_ownership",
    "term",
]


def parse_term_months(value: object) -> float:
    if pd.isna(value):
        return np.nan
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return float(digits) if digits else np.nan


def apply_capping(data: pd.DataFrame, cap_rules: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Create capped numeric columns using training-time percentile rules."""
    result = data.copy()
    rules = DEFAULT_CAP_RULES if cap_rules is None else cap_rules

    for col, limits in rules.items():
        if col in result.columns:
            result[f"{col}_capped"] = result[col].clip(
                lower=limits["lower_p01"],
                upper=limits["upper_p99"],
            )
    return result


def engineer_features(data: pd.DataFrame, cap_rules: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Apply the feature logic used in the model development notebooks."""
    result = data.copy()

    if "issue_d" in result.columns:
        result["issue_d"] = pd.to_datetime(result["issue_d"], errors="coerce")

    if {"issue_d", "earliest_cr_line"}.issubset(result.columns):
        result["earliest_cr_line"] = pd.to_datetime(result["earliest_cr_line"], errors="coerce")
        result["credit_history_months"] = (
            (result["issue_d"].dt.year - result["earliest_cr_line"].dt.year) * 12
            + (result["issue_d"].dt.month - result["earliest_cr_line"].dt.month)
        )
        result["credit_history_years"] = result["credit_history_months"] / 12
        result["credit_history_missing_flag"] = result["earliest_cr_line"].isna().astype(int)

    if "term" in result.columns:
        result["term_months"] = result["term"].apply(parse_term_months)
        result["is_60m_term"] = (result["term_months"] == 60).astype(int)

    if {"fico_range_low", "fico_range_high"}.issubset(result.columns):
        result["fico_mid"] = (result["fico_range_low"] + result["fico_range_high"]) / 2
    elif "fico_range_high" in result.columns:
        result["fico_mid"] = result["fico_range_high"] - 5

    if {"loan_amnt", "annual_inc"}.issubset(result.columns):
        result["loan_to_income"] = np.where(
            result["annual_inc"] > 0,
            result["loan_amnt"] / result["annual_inc"],
            np.nan,
        )
        result["annual_inc_missing_flag"] = result["annual_inc"].isna().astype(int)

    if {"installment", "annual_inc"}.issubset(result.columns):
        result["installment_to_monthly_income"] = np.where(
            result["annual_inc"] > 0,
            result["installment"] / (result["annual_inc"] / 12),
            np.nan,
        )

    if "revol_util" in result.columns:
        result["revol_util_missing_flag"] = result["revol_util"].isna().astype(int)
        result["high_revol_util_flag"] = (result["revol_util"] >= 80).astype(int)

    if "dti" in result.columns:
        result["dti_missing_flag"] = result["dti"].isna().astype(int)
        result["high_dti_flag"] = (result["dti"] >= 35).astype(int)

    flag_sources = {
        "delinq_2yrs": "has_recent_delinq_flag",
        "inq_last_6mths": "has_recent_inquiry_flag",
        "pub_rec": "has_public_record_flag",
        "pub_rec_bankruptcies": "has_bankruptcy_flag",
        "mort_acc": "has_mortgage_account_flag",
    }
    for source, flag in flag_sources.items():
        if source in result.columns:
            result[flag] = (result[source].fillna(0) > 0).astype(int)

    if "emp_length" in result.columns:
        emp = result["emp_length"].astype(str).str.lower()
        result["emp_length_missing_flag"] = result["emp_length"].isna().astype(int)
        result["emp_length_num"] = np.nan
        result.loc[emp.str.contains("< 1", na=False), "emp_length_num"] = 0
        result.loc[emp.str.contains("10+", regex=False, na=False), "emp_length_num"] = 10
        extracted = emp.str.extract(r"(\d+)")[0]
        mask = result["emp_length_num"].isna() & extracted.notna()
        result.loc[mask, "emp_length_num"] = extracted[mask].astype(float)

    if "issue_d" in result.columns:
        result["issue_year_fe"] = result["issue_d"].dt.year
        result["issue_quarter_fe"] = result["issue_d"].dt.quarter

    return apply_capping(result, cap_rules=cap_rules)
