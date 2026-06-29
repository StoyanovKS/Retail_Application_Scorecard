"""Command line batch prediction entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scorecard.model import ScorecardModel


def predict_file(input_path: str | Path, output_path: str | Path, config_path: str | Path | None = None) -> pd.DataFrame:
    """Score a CSV file and write predictions to CSV."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    data = pd.read_csv(input_path)
    scorer = ScorecardModel.from_files(config_path=config_path)
    predictions = scorer.predict(data)
    output = pd.concat([data.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Score loan applications with the credit scorecard.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--config", default=None, help="Optional config YAML path.")
    args = parser.parse_args()

    predict_file(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
