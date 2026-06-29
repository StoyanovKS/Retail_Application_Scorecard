# Retail Application Credit Scorecard

The data used in this project is from the Kaggle dataset **Historical Loan Records with Default Status**.

## Overview

This project develops an end-to-end **Application Credit Scoring Model** using historical loan data. The objective is to predict the probability of loan default and build an interpretable scorecard that can support credit risk assessment and lending decisions.

The project follows a complete credit risk modelling workflow, including:

* Data inspection and target review
* Data quality and leakage assessment
* Exploratory data analysis
* Time-based model development framework
* Feature engineering
* WOE binning and IV analysis
* Logistic Regression scorecard development
* Challenger model benchmarking
* Validation and stability assessment
* Score scaling and cutoff analysis
* Model governance overview

The final selected model is a **WOE-transformed Logistic Regression scorecard**.

---

## Project Structure

```text
credit_scoring_loans/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_target_definition_and_leakage.ipynb
│   ├── 03_eda_and_univariate_analysis.ipynb
│   ├── 04_time_split_and_sampling_strategy.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_woe_binning_iv_analysis.ipynb
│   ├── 07_logistic_regression_scorecard.ipynb
│   ├── 08_model_benchmarking.ipynb
│   ├── 09_validation_and_stability.ipynb
│   ├── 10_score_scaling_and_cutoffs_short.ipynb
│   └── 11_model_governance.ipynb
│
└── README.md
```

---

# Notebook Descriptions

## 01_data_inspection.ipynb

Initial data exploration and quality assessment.

Main activities:

* Dataset structure review
* Variable type identification
* Missing value analysis
* Descriptive statistics
* Target distribution review
* Date field inspection
* Data quality assessment

Outputs:

* Data inspection report
* Variable inventory
* Missing value summary

---

## 02_target_definition_and_leakage.ipynb

Target review and leakage assessment.

Main activities:

* Target variable validation
* Good / bad definition review
* Maturity bias analysis
* Leakage detection
* Vintage-level analysis
* Creation of modelling dataset

Outputs:

* Target definition report
* Leakage assessment
* Modelling population dataset

---

## 03_eda_and_univariate_analysis.ipynb

Exploratory data analysis and univariate risk assessment.

Main activities:

* Variable distribution analysis
* Risk segmentation by variable
* Vintage performance analysis
* Bad rate trends
* Business interpretation of key predictors

Outputs:

* EDA report
* Univariate risk analysis
* Variable-level charts

---

## 04_time_split_and_sampling_strategy.ipynb

Definition of the model development framework.

Main activities:

* Time-based train / validation / OOT split
* Sample construction
* Vintage allocation review
* Development and validation population creation

Outputs:

* Train dataset
* Validation dataset
* Out-of-time test dataset

---

## 05_feature_engineering.ipynb

Feature preparation and engineering.

Main activities:

* Derived variable creation
* Missing value treatment
* Outlier treatment
* Variable transformations
* Candidate feature selection

Outputs:

* Engineered datasets
* Candidate modelling features

---

## 06_woe_binning_iv_analysis.ipynb

WOE transformation and variable selection process.

Main activities:

* Optimal binning
* Weight of Evidence transformation
* Information Value calculation
* Correlation analysis
* Business review of candidate variables
* Final variable selection

Outputs:

* WOE-transformed datasets
* IV report
* Correlation analysis
* Final modelling variables

---

## 07_logistic_regression_scorecard.ipynb

Development of the final scorecard model.

Main activities:

* Logistic Regression estimation
* Coefficient review
* Statistical significance assessment
* Multicollinearity checks
* Model performance evaluation
* Scorecard generation

Outputs:

* Final scorecard model
* Predicted PDs
* Performance metrics
* Score scaling framework

---

## 08_model_benchmarking.ipynb

Benchmark comparison against challenger models.

Main activities:

* Logistic Regression benchmark
* Alternative sampling framework comparison
* XGBoost challenger model
* Performance comparison
* Interpretability versus performance assessment

Outputs:

* Benchmarking report
* Challenger model comparison
* Performance ranking

---

## 09_validation_and_stability.ipynb

Independent-style validation and stability assessment.

Main activities:

* Train / Validation / OOT comparison
* AUROC, Gini and KS evaluation
* Calibration assessment
* Population Stability Index (PSI)
* Characteristic stability analysis
* Vintage performance review

Outputs:

* Validation report
* Stability analysis
* Monitoring metrics

---

## 10_score_scaling_and_cutoffs_short.ipynb

Business interpretation of model scores and decision thresholds.

Main activities:

* Score scaling review
* Score decile analysis
* PD cutoff scenarios
* Approval strategy assessment
* Portfolio quality trade-off analysis

Outputs:

* Cutoff analysis
* Approval strategy scenarios
* Scorecard business interpretation

---

## 11_model_governance.ipynb

Governance overview covering model ownership, validation, monitoring, documentation, implementation controls and model risk management considerations.

---

# Modelling Approach

The final modelling framework consists of:

1. Time-based development and validation split
2. Feature engineering and data preparation
3. WOE transformation and IV-based screening
4. Correlation and business-rule review
5. Logistic Regression scorecard development
6. Challenger model benchmarking
7. Validation and stability assessment
8. Score scaling and cutoff analysis
9. Governance and monitoring considerations

---

# Key Techniques Used

* Credit Risk Modelling
* Logistic Regression
* Weight of Evidence (WOE)
* Information Value (IV)
* Scorecard Development
* Population Stability Index (PSI)
* AUROC / Gini / KS
* Calibration Analysis
* Time-Based Validation
* Model Benchmarking
* XGBoost
* Score Scaling
* Cutoff Analysis

---

# Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Statsmodels
* OptBinning
* XGBoost
* Matplotlib
* OpenPyXL
* Jupyter Notebook

---

# Conclusion

The final selected model is a WOE-transformed Logistic Regression scorecard supported by validation, stability analysis, challenger benchmarking and score scaling.

## Model Performance

| Sample | AUROC | Gini | KS |
| --- | ---: | ---: | ---: |
| Train | 0.692 | 0.385 | 0.278 |
| Validation | 0.672 | 0.343 | 0.245 |
| OOT | 0.667 | 0.334 | 0.247 |

The final scorecard demonstrates stable performance across Train, Validation and Out-of-Time samples with limited evidence of overfitting.

---

# Production Python Package

The project is prepared to be extended into a production-ready Python package under `src/scorecard`.
The purpose of this package is to separate reusable model logic from exploratory notebooks and make the credit scoring model easier to test, deploy and maintain.

The package follows the same modelling pipeline developed in the notebooks:

1. Input loan application data
2. Apply feature engineering
3. Apply trained WOE transformations
4. Generate model-ready variables
5. Predict probability of default
6. Convert predicted PD into a credit score
7. Return scoring outputs for business use

---

# Package Structure

```text
src/
`-- scorecard/
    |-- __init__.py
    |-- feature_engineering.py
    |-- preprocessing.py
    |-- woe.py
    |-- scoring.py
    |-- predict.py
    |-- model.py
    `-- utils.py
```

---

# Module Responsibilities

## `feature_engineering.py`

Contains reusable feature engineering logic used before model scoring.

This module should reproduce the same transformations developed in the notebooks, including:

* Term parsing
* Credit history calculations
* Loan-to-income calculation
* Missing value flags
* Risk flags
* Outlier capping rules
* Engineered variables required by the scorecard

This ensures that production scoring uses the same feature definitions as model development.

## `preprocessing.py`

Combines feature engineering and WOE preparation into a single preprocessing step.

This module prepares raw loan application records into the final model input format expected by the trained Logistic Regression model.

## `woe.py`

Handles Weight of Evidence transformation.

This module loads and applies the trained WOE binning objects saved during model development. It ensures that new production records are transformed using the same bins and WOE values used during training.

## `scoring.py`

Contains score scaling logic.

This module converts predicted probability of default into scorecard points using the score scaling parameters:

* Base score
* Base odds
* Points to double the odds

It also provides utility functions for converting between PD and score.

## `model.py`

Contains the main production model wrapper.

This module loads the trained Logistic Regression model, WOE bins and configuration, then exposes a clean scoring interface.

Typical use:

```python
from scorecard.model import ScorecardModel

model = ScorecardModel.from_files()
predictions = model.predict(applications)
```

The output contains predicted default probability and credit score.

## `predict.py`

Provides a batch scoring entry point.

This module can be used to score a CSV file from the command line and write the scored output to disk.

Example:

```bash
scorecard-predict --input new_applications.csv --output scored_applications.csv
```

## `utils.py`

Contains shared helper functions for:

* Loading model artifacts
* Loading configuration files
* Resolving project paths
* Saving and loading pickle files

## `__init__.py`

Defines the public package interface and makes key package functions available through `scorecard`.

---

# Model Artifacts

Production model artifacts are stored in the `models/` directory.

```text
models/
|-- logistic_model.pkl
|-- scorecard_points.csv
`-- woe_bins.pkl
```

## `logistic_model.pkl`

The trained WOE Logistic Regression model.

## `woe_bins.pkl`

The fitted WOE binning objects used to transform raw and engineered variables.

## `scorecard_points.csv`

The scorecard points table used for interpretation, documentation and business review.

---

# Configuration

The package uses `configs/config.yaml` to store production settings.

```text
configs/
`-- config.yaml
```

The configuration includes:

* Model artifact paths
* Final model feature list
* Score scaling parameters
* Target definition
* Capping rules

This keeps production settings outside the Python code and makes the package easier to maintain.

---

# Installation

For local development, install the package in editable mode:

```bash
pip install -e ".[dev]"
```

This allows changes in `src/scorecard` to be immediately available without reinstalling the package.

To install required dependencies only:

```bash
pip install -r requirements.txt
```

---

# Running Tests

Tests are stored in the `tests/` directory.

```text
tests/
|-- test_scoring.py
|-- test_feature_engineering.py
`-- test_woe.py
```

Run tests with:

```bash
python -m pytest
```

The tests validate that:

* Feature engineering creates expected fields
* WOE transformation works with trained bins
* Score scaling behaves correctly
* Model artifacts can be loaded
* The scoring pipeline can produce predictions

---

# Scoring From a CSV File

The production package supports batch scoring from a CSV file. The expected workflow is to place input files in the `inputs/` folder and write scored files to the `outputs/` folder.

```text
inputs/
`-- sample.csv

outputs/
`-- scored_sample.csv
```

The repository includes an example input file:

```text
inputs/sample.csv
```

This file contains two example loan application records with the minimum fields required by the current scorecard pipeline.

## Required Input Columns

The scoring CSV should contain the following columns:

| Column | Purpose |
| --- | --- |
| `issue_d` | Loan issue or origination date used for date-based feature engineering |
| `earliest_cr_line` | Earliest credit line date used to calculate credit history length |
| `term` | Loan term, for example `36 months` or `60 months` |
| `loan_amnt` | Requested loan amount |
| `annual_inc` | Applicant annual income |
| `dti` | Debt-to-income ratio |
| `revol_util` | Revolving credit utilization |
| `inq_last_6mths` | Number of recent credit inquiries |
| `acc_open_past_24mths` | Number of accounts opened in the past 24 months |
| `mort_acc` | Number of mortgage accounts |
| `purpose` | Loan purpose |
| `home_ownership` | Applicant home ownership status |

The package will use these fields to create engineered variables, apply WOE transformation, predict probability of default and calculate the final score.

## Run Scoring

Run the command from the project root:

```bash
scorecard-predict --input inputs/sample.csv --output outputs/scored_sample.csv
```

If the package has not been installed as a command line tool yet, install it first:

```bash
pip install -e ".[dev]"
```

Alternatively, the same scoring logic can be called from Python:

```python
from scorecard.predict import predict_file

predict_file(
    input_path="inputs/sample.csv",
    output_path="outputs/scored_sample.csv",
)
```

## Expected Output

The scored file will be created here:

```text
outputs/scored_sample.csv
```

The output file keeps the original input columns and appends two scoring columns:

| Output column | Meaning |
| --- | --- |
| `pd_pred` | Predicted probability of default |
| `score` | Credit score scaled from the predicted PD |

Example output structure:

```text
issue_d,earliest_cr_line,term,loan_amnt,...,home_ownership,pd_pred,score
2017-01-01,2010-01-01,36 months,10000,...,MORTGAGE,0.127798,522.443703
2018-03-01,2012-05-01,60 months,25000,...,RENT,0.467414,393.319247
```

For production batch scoring, place any new application file in `inputs/`, choose an output name under `outputs/`, and run the same command with the new paths.


--- 


# Production Scoring Pipeline

The intended production scoring flow is:

```text
Raw application data
        |
        v
Feature engineering
        |
        v
WOE transformation
        |
        v
Model input frame
        |
        v
Logistic Regression PD prediction
        |
        v
Score scaling
        |
        v
PD and credit score output
```

This structure separates research from production logic. The notebooks remain the modelling and analysis layer, while `src/scorecard` becomes the reusable scoring package for deployment, testing and future application development.
