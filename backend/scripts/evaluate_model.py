import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from app.config import get_settings


def evaluate(features_csv: Path, model_path: Path) -> None:
    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}. Run train_baseline.py first.")
    if not features_csv.exists():
        raise SystemExit(f"Features CSV not found: {features_csv}")

    bundle = joblib.load(model_path)
    frame = pd.read_csv(features_csv)
    if "owner_label" not in frame.columns:
        raise SystemExit("Features CSV must include owner_label for evaluation.")

    feature_names = bundle["feature_names"]
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise SystemExit(f"Features CSV is missing model features: {missing}")

    x = frame[feature_names].fillna(0.0)
    y = frame["owner_label"].astype(str)
    predictions = bundle["model"].predict(x)
    print(classification_report(y, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y, predictions, labels=bundle["labels"]))
    print("Labels:", bundle["labels"])


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate a trained DogBridge baseline model.")
    parser.add_argument("--features-csv", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--model-path", type=Path, default=settings.model_path)
    args = parser.parse_args()
    evaluate(args.features_csv, args.model_path)


if __name__ == "__main__":
    main()

