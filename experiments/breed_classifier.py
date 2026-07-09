import argparse
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", message="Unknown solver options: iprint")

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


def balanced_sample(frame: pd.DataFrame, label_column: str, max_examples_per_label: int) -> pd.DataFrame:
    samples = [
        group.sample(n=min(max_examples_per_label, len(group)), random_state=42)
        for _, group in frame.groupby(label_column)
    ]
    return pd.concat(samples, ignore_index=True)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root(),
        )
        return result.stdout.strip()
    except Exception:
        return None


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

    f0 = librosa.yin(y, fmin=50, fmax=2500, sr=sr)
    voiced_f0 = f0[np.isfinite(f0)]
    features.update(stats("pitch_f0", voiced_f0))

    features.update(stats("rms", librosa.feature.rms(y=y)[0]))
    features.update(stats("zero_crossing_rate", librosa.feature.zero_crossing_rate(y)[0]))
    features.update(stats("spectral_centroid", librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
    features.update(stats("spectral_bandwidth", librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))
    features.update(stats("spectral_rolloff", librosa.feature.spectral_rolloff(y=y, sr=sr)[0]))
    features.update(stats("spectral_flatness", librosa.feature.spectral_flatness(y=y)[0]))
    return features


def build_feature_frame(
    manifest_csv: Path,
    features_csv: Path | None = None,
    rebuild_features: bool = False,
    checkpoint_every: int = 500,
    max_examples_per_breed: int | None = None,
) -> pd.DataFrame:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")

    manifest = pd.read_csv(manifest_csv)
    required = {"audio_file_path", LABEL_COLUMN}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest CSV is missing required columns: {sorted(missing)}")
    manifest = manifest.assign(**{LABEL_COLUMN: manifest[LABEL_COLUMN].astype(str).str.strip().str.lower()})
    if max_examples_per_breed is not None:
        manifest = balanced_sample(manifest, LABEL_COLUMN, max_examples_per_breed)

    rows = []
    processed_paths = set()
    if features_csv and features_csv.exists() and not rebuild_features:
        existing = pd.read_csv(features_csv)
        if "audio_file_path" in existing.columns:
            rows = existing.to_dict("records")
            processed_paths = set(existing["audio_file_path"].astype(str))
            print(f"Loaded {len(rows)} cached feature rows from {features_csv}")

    total = len(manifest)
    for index, item in manifest.iterrows():
        breed = str(item.get(LABEL_COLUMN, "")).strip().lower()
        if not breed or breed == "nan":
            continue

        audio_path = Path(str(item["audio_file_path"]))
        if str(audio_path) in processed_paths:
            continue
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
            if features_csv and checkpoint_every > 0 and len(rows) % checkpoint_every == 0:
                features_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(features_csv, index=False)
                print(f"Extracted {len(rows)} feature rows ({index + 1}/{total} manifest rows)")
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


def feature_columns_for(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def make_model(model_type: str):
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1)
    if model_type == "extra_trees":
        return ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1)
    if model_type == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=42, class_weight="balanced", n_jobs=1),
        )
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=300, random_state=42, class_weight="balanced")
    raise ValueError(f"Unsupported model type: {model_type}")


def top_k_scores(y_true: pd.Series, probabilities: np.ndarray, labels: list[str]) -> dict[str, float]:
    scores = {}
    for k in (2, 3):
        if len(labels) >= k:
            scores[f"top_{k}_accuracy"] = float(top_k_accuracy_score(y_true, probabilities, k=k, labels=labels))
    return scores


def write_evaluation_outputs(
    report_dir: Path,
    report_text: str,
    metrics: dict,
    labels: list[str],
    y_test: pd.Series,
    predictions: np.ndarray,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "breed_classification_report.txt").write_text(report_text, encoding="utf-8")
    (report_dir / "breed_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.to_csv(report_dir / "breed_confusion_matrix.csv")


def train_breed_classifier(
    frame: pd.DataFrame,
    model_path: Path,
    min_examples_per_breed: int,
    max_examples_per_breed: int | None = None,
    split_strategy: str = "dog-grouped",
    model_type: str = "random_forest",
    test_size: float = 0.25,
    report_dir: Path | None = None,
    refit_full_data: bool = True,
) -> None:
    counts = frame[LABEL_COLUMN].value_counts()
    usable_breeds = counts[counts >= min_examples_per_breed].index
    frame = frame[frame[LABEL_COLUMN].isin(usable_breeds)].copy()
    if max_examples_per_breed is not None:
        frame = balanced_sample(frame, LABEL_COLUMN, max_examples_per_breed)

    if frame[LABEL_COLUMN].nunique() < 2:
        raise SystemExit(
            "Training requires at least two breeds with enough examples. "
            f"Current minimum per breed is {min_examples_per_breed}."
        )

    feature_columns = feature_columns_for(frame)
    if not feature_columns:
        raise SystemExit("No numeric acoustic feature columns found.")

    x = frame[feature_columns].fillna(0.0)
    y = frame[LABEL_COLUMN].astype(str)
    groups = frame["dog_id"].astype(str) if "dog_id" in frame.columns else None

    if split_strategy == "dog-grouped":
        if groups is None or groups.nunique() < 2:
            raise SystemExit("Dog-grouped splitting requires at least two dog_id groups.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_index, test_index = next(splitter.split(x, y, groups))
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        train_groups = sorted(groups.iloc[train_index].unique())
        test_groups = sorted(groups.iloc[test_index].unique())
    elif split_strategy == "random":
        stratify = y if y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
        train_groups = sorted(groups.loc[x_train.index].unique()) if groups is not None else []
        test_groups = sorted(groups.loc[x_test.index].unique()) if groups is not None else []
    else:
        raise SystemExit("--split-strategy must be either 'dog-grouped' or 'random'.")

    model = make_model(model_type)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = list(model.classes_)
    probabilities = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None

    report_text = classification_report(y_test, predictions, zero_division=0)
    report_dict = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    metrics = {
        "model_version": f"dogbridge_breed_{model_type}_v2",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "split_strategy": split_strategy,
        "model_type": model_type,
        "test_size": test_size,
        "refit_full_data": refit_full_data,
        "row_count": int(len(frame)),
        "feature_count": int(len(feature_columns)),
        "labels": labels,
        "breed_counts": {label: int(count) for label, count in y.value_counts().items()},
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_dogs": int(len(train_groups)),
        "test_dogs": int(len(test_groups)),
        "train_dog_ids": train_groups,
        "test_dog_ids": test_groups,
        "accuracy": float(report_dict["accuracy"]),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    }
    if probabilities is not None:
        metrics.update(top_k_scores(y_test, probabilities, labels))

    print("Breed counts used for training:")
    print(y.value_counts().to_string())
    print()
    print(f"Split strategy: {split_strategy}")
    print(f"Train rows: {len(x_train)} | Test rows: {len(x_test)}")
    if groups is not None:
        print(f"Train dogs: {len(train_groups)} | Test dogs: {len(test_groups)}")
    print()
    print(report_text)
    if "top_2_accuracy" in metrics:
        print(f"Top-2 accuracy: {metrics['top_2_accuracy']:.3f}")
    if "top_3_accuracy" in metrics:
        print(f"Top-3 accuracy: {metrics['top_3_accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")

    if report_dir:
        write_evaluation_outputs(report_dir, report_text, metrics, labels, y_test, predictions)
        print(f"Wrote evaluation outputs to {report_dir}")

    final_model = make_model(model_type)
    if refit_full_data:
        final_model.fit(x, y)
    else:
        final_model = model

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "labels": list(final_model.classes_),
            "feature_names": feature_columns,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "model_version": metrics["model_version"],
            "metrics": metrics,
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
    parser.add_argument("--report-dir", type=Path, default=root / "backend" / "data" / "models" / "reports" / "breed_classifier")
    parser.add_argument("--min-examples-per-breed", type=int, default=5)
    parser.add_argument("--max-examples-per-breed", type=int, default=None)
    parser.add_argument("--split-strategy", choices=["dog-grouped", "random"], default="dog-grouped")
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "extra_trees", "logistic_regression", "hist_gradient_boosting"],
        default="logistic_regression",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--no-refit-full-data", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--predict-audio", type=Path, default=None)
    args = parser.parse_args()

    if args.predict_audio:
        predict_breed(args.predict_audio, args.model_path)
        return

    if args.min_examples_per_breed < 1:
        raise SystemExit("--min-examples-per-breed must be at least 1.")
    if args.max_examples_per_breed is not None and args.max_examples_per_breed < 1:
        raise SystemExit("--max-examples-per-breed must be at least 1 when provided.")
    if not 0.0 < args.test_size < 1.0:
        raise SystemExit("--test-size must be between 0 and 1.")

    frame = build_feature_frame(
        args.manifest_csv,
        args.features_csv,
        args.rebuild_features,
        args.checkpoint_every,
        args.max_examples_per_breed,
    )
    train_breed_classifier(
        frame,
        args.model_path,
        args.min_examples_per_breed,
        args.max_examples_per_breed,
        args.split_strategy,
        args.model_type,
        args.test_size,
        args.report_dir,
        not args.no_refit_full_data,
    )


if __name__ == "__main__":
    main()
