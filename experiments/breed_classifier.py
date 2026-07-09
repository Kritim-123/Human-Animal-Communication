import argparse
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


TARGET_SAMPLE_RATE = 22050
LABEL_COLUMN = "breed"
NON_FEATURE_COLUMNS = {
    "dog_id",
    "audio_file_name",
    "folder_name",
    "audio_file_path",
    "breed",
    "sex",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        clean = np.array([0.0])
    return {
        f"{prefix}_mean": float(np.mean(clean)),
        f"{prefix}_std": float(np.std(clean)),
        f"{prefix}_min": float(np.min(clean)),
        f"{prefix}_max": float(np.max(clean)),
    }


def extract_features(audio_path: Path) -> dict[str, float]:
    y, sr = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {audio_path}")

    duration_seconds = float(librosa.get_duration(y=y, sr=sr))
    features: dict[str, float] = {"duration_seconds": duration_seconds}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for index, row in enumerate(mfcc, start=1):
        features.update(stats(f"mfcc_{index}", row))

    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=50, fmax=2500, sr=sr)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    features.update(stats("pitch_f0", voiced_f0))
    features["voiced_fraction"] = float(np.mean(voiced_flag)) if voiced_flag is not None and voiced_flag.size else 0.0
    features["voiced_probability_mean"] = float(np.nanmean(voiced_prob)) if voiced_prob is not None else 0.0

    features.update(stats("rms", librosa.feature.rms(y=y)[0]))
    features.update(stats("zero_crossing_rate", librosa.feature.zero_crossing_rate(y)[0]))
    features.update(stats("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
    features.update(stats("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))
    features.update(stats("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
    features.update(stats("spectral_flatness", librosa.feature.spectral_flatness(y=y)[0]))
    return features


def build_feature_frame(manifest_csv: Path, features_csv: Path | None = None) -> pd.DataFrame:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")

    manifest = pd.read_csv(manifest_csv)
    required = {"audio_file_path", LABEL_COLUMN}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest CSV is missing required columns: {sorted(missing)}")

    rows = []
    for _, item in manifest.iterrows():
        breed = str(item.get(LABEL_COLUMN, "")).strip().lower()
        if not breed or breed == "nan":
            continue

        audio_path = Path(str(item["audio_file_path"]))
        if not audio_path.exists():
            print(f"Skipping missing file: {audio_path}")
            continue

        try:
            base_row = {
                "dog_id": item.get("dog_id", ""),
                "audio_file_name": item.get("audio_file_name", audio_path.name),
                "folder_name": item.get("folder_name", audio_path.parent.name),
                "audio_file_path": str(audio_path),
                "breed": breed,
                "sex": item.get("sex", ""),
            }
            rows.append({**base_row, **extract_features(audio_path)})
        except Exception as exc:
            print(f"Skipping {audio_path}: {exc}")

    if not rows:
        raise SystemExit("No usable labeled audio rows found. Check paths and breed labels in the manifest.")

    frame = pd.DataFrame(rows)
    if features_csv:
        features_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(features_csv, index=False)
        print(f"Wrote extracted breed features to {features_csv}")
    return frame


def train_breed_classifier(frame: pd.DataFrame, model_path: Path, min_examples_per_breed: int) -> None:
    counts = frame[LABEL_COLUMN].value_counts()
    usable_breeds = counts[counts >= min_examples_per_breed].index
    frame = frame[frame[LABEL_COLUMN].isin(usable_breeds)].copy()

    if frame[LABEL_COLUMN].nunique() < 2:
        raise SystemExit(
            "Training requires at least two breeds with enough examples. "
            f"Current minimum per breed is {min_examples_per_breed}."
        )

    feature_columns = [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_columns:
        raise SystemExit("No numeric acoustic feature columns found.")

    x = frame[feature_columns].fillna(0.0)
    y = frame[LABEL_COLUMN].astype(str)
    stratify = y if y.value_counts().min() >= 2 else None

    test_size = 0.25 if len(frame) >= 20 else 0.35
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    model = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("Breed counts used for training:")
    print(y.value_counts().to_string())
    print()
    print(classification_report(y_test, predictions, zero_division=0))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": list(model.classes_),
            "feature_names": feature_columns,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "model_version": "dogbridge_breed_random_forest_v1",
        },
        model_path,
    )
    print(f"Saved breed classifier to {model_path}")


def predict_breed(audio_path: Path, model_path: Path) -> None:
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Train the classifier first.")
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    labels = bundle["labels"]

    features = pd.DataFrame([extract_features(audio_path)])
    x = features.reindex(columns=feature_names, fill_value=0.0).fillna(0.0)

    predicted_label = str(model.predict(x)[0])
    print(f"Predicted breed: {predicted_label}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)[0]
        top = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)[:5]
        print("Top predictions:")
        for label, probability in top:
            print(f"  {label}: {probability:.3f}")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Train a dog breed classifier from a dog audio manifest CSV.")
    parser.add_argument("--manifest-csv", type=Path, default=root / "backend" / "data" / "processed" / "dog_audio_manifest.csv")
    parser.add_argument("--features-csv", type=Path, default=root / "backend" / "data" / "processed" / "breed_features.csv")
    parser.add_argument("--model-path", type=Path, default=root / "backend" / "data" / "models" / "breed_classifier.joblib")
    parser.add_argument("--min-examples-per-breed", type=int, default=5)
    parser.add_argument("--predict-audio", type=Path, default=None)
    args = parser.parse_args()

    if args.predict_audio:
        predict_breed(args.predict_audio, args.model_path)
        return

    if args.min_examples_per_breed < 1:
        raise SystemExit("--min-examples-per-breed must be at least 1.")

    frame = build_feature_frame(args.manifest_csv, args.features_csv)
    train_breed_classifier(frame, args.model_path, args.min_examples_per_breed)


if __name__ == "__main__":
    main()
