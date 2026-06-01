import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.database import SessionLocal
from app.models import AudioClip
from scripts.extract_features import extract_feature_dict


LABEL_COLUMN = "owner_label"
NON_FEATURE_COLUMNS = {
    "id",
    "clip_id",
    "dog_id",
    "file_path",
    "duration_seconds",
    "recorded_at",
    "location_context",
    "situation_context",
    "owner_label",
    "outcome_label",
    "prediction_label",
    "prediction_confidence",
    "confirmed_correct",
    "notes",
    "created_at",
}


def _features_from_database() -> pd.DataFrame:
    db = SessionLocal()
    try:
        clips = db.query(AudioClip).filter(AudioClip.owner_label.isnot(None)).all()
        rows = []
        for clip in clips:
            path = Path(clip.file_path)
            if not path.exists():
                print(f"Skipping clip {clip.id}: audio file missing at {path}")
                continue
            try:
                rows.append(
                    {
                        "clip_id": clip.id,
                        "file_path": str(path),
                        LABEL_COLUMN: clip.owner_label,
                        **extract_feature_dict(path),
                    }
                )
            except Exception as exc:
                print(f"Skipping clip {clip.id}: {exc}")
        return pd.DataFrame(rows)
    finally:
        db.close()


def _load_training_frame(features_csv: Path | None) -> pd.DataFrame:
    if features_csv:
        if not features_csv.exists():
            raise SystemExit(f"Features CSV not found: {features_csv}")
        return pd.read_csv(features_csv)
    return _features_from_database()


def train(features_csv: Path | None, model_path: Path) -> None:
    frame = _load_training_frame(features_csv)
    if frame.empty:
        raise SystemExit(
            "No usable labeled clips found. Upload real audio with owner labels or pass --features-csv with extracted features."
        )
    if LABEL_COLUMN not in frame.columns:
        raise SystemExit(f"Training data must include '{LABEL_COLUMN}'.")

    frame = frame.dropna(subset=[LABEL_COLUMN])
    feature_columns = [column for column in frame.columns if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])]
    if not feature_columns:
        raise SystemExit("No numeric feature columns found. Run extract_features.py first.")
    if frame[LABEL_COLUMN].nunique() < 2:
        raise SystemExit("Training requires at least two intent labels. Collect more labeled clips.")
    if len(frame) < 6:
        raise SystemExit("Training requires at least 6 labeled clips for a minimally useful baseline.")

    x = frame[feature_columns].fillna(0.0)
    y = frame[LABEL_COLUMN].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=stratify)

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred, zero_division=0))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": list(model.classes_),
            "feature_names": feature_columns,
            "model_version": "dogbridge_baseline_random_forest_v1",
        },
        model_path,
    )
    print(f"Saved baseline model to {model_path}")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train DogBridge baseline likely-intent classifier.")
    parser.add_argument("--features-csv", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=settings.model_path)
    args = parser.parse_args()
    train(args.features_csv, args.model_path)


if __name__ == "__main__":
    main()
