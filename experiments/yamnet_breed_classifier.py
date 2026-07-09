import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
TARGET_SAMPLE_RATE = 16000
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


def balanced_sample(frame: pd.DataFrame, label_column: str, max_examples_per_label: int) -> pd.DataFrame:
    samples = [
        group.sample(n=min(max_examples_per_label, len(group)), random_state=42)
        for _, group in frame.groupby(label_column)
    ]
    return pd.concat(samples, ignore_index=True)


def load_manifest(manifest_csv: Path, max_examples_per_breed: int | None) -> pd.DataFrame:
    if not manifest_csv.exists():
        raise SystemExit(f"Manifest CSV not found: {manifest_csv}")

    manifest = pd.read_csv(manifest_csv)
    required = {"audio_file_path", LABEL_COLUMN}
    missing = required - set(manifest.columns)
    if missing:
        raise SystemExit(f"Manifest CSV is missing required columns: {sorted(missing)}")

    manifest = manifest.assign(**{LABEL_COLUMN: manifest[LABEL_COLUMN].astype(str).str.strip().str.lower()})
    manifest = manifest[manifest[LABEL_COLUMN].notna()]
    if max_examples_per_breed is not None:
        manifest = balanced_sample(manifest, LABEL_COLUMN, max_examples_per_breed)
    return manifest.reset_index(drop=True)


def extract_yamnet_embedding(audio_path: Path, yamnet_model) -> dict[str, float]:
    waveform, _ = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {audio_path}")

    waveform = waveform.astype(np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1.0:
        waveform = waveform / peak

    _, embeddings, _ = yamnet_model(tf.convert_to_tensor(waveform, dtype=tf.float32))
    embedding_np = embeddings.numpy()
    if embedding_np.ndim != 2 or embedding_np.shape[0] == 0:
        raise ValueError(f"YAMNet produced no embeddings for {audio_path}")

    pooled_mean = embedding_np.mean(axis=0)
    pooled_std = embedding_np.std(axis=0)

    features = {"duration_seconds": float(len(waveform) / TARGET_SAMPLE_RATE)}
    for index, value in enumerate(pooled_mean):
        features[f"yamnet_mean_{index}"] = float(value)
    for index, value in enumerate(pooled_std):
        features[f"yamnet_std_{index}"] = float(value)
    return features


def build_embedding_frame(
    manifest_csv: Path,
    embeddings_csv: Path,
    max_examples_per_breed: int | None,
    rebuild_embeddings: bool,
    checkpoint_every: int,
) -> pd.DataFrame:
    manifest = load_manifest(manifest_csv, max_examples_per_breed)
    rows = []
    processed_paths = set()

    if embeddings_csv.exists() and not rebuild_embeddings:
        existing = pd.read_csv(embeddings_csv)
        if "audio_file_path" in existing.columns:
            rows = existing.to_dict("records")
            processed_paths = set(existing["audio_file_path"].astype(str))
            print(f"Loaded {len(rows)} cached YAMNet embedding rows from {embeddings_csv}")

    yamnet_model = hub.load(YAMNET_URL)
    total = len(manifest)
    for index, item in manifest.iterrows():
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
                "breed": str(item.get(LABEL_COLUMN, "")).strip().lower(),
                "sex": item.get("sex", ""),
            }
            rows.append({**base_row, **extract_yamnet_embedding(audio_path, yamnet_model)})
            if checkpoint_every > 0 and len(rows) % checkpoint_every == 0:
                embeddings_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(embeddings_csv, index=False)
                print(f"Extracted {len(rows)} YAMNet rows ({index + 1}/{total} manifest rows)")
        except Exception as exc:
            print(f"Skipping {audio_path}: {exc}")

    if not rows:
        raise SystemExit("No usable YAMNet embedding rows found.")

    frame = pd.DataFrame(rows)
    embeddings_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(embeddings_csv, index=False)
    print(f"Wrote YAMNet embeddings to {embeddings_csv}")
    return frame


def feature_columns_for(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def make_model(model_type: str):
    if model_type == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=42, class_weight="balanced"),
        )
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1)
    raise ValueError(f"Unsupported model type: {model_type}")


def top_k_scores(y_true: pd.Series, probabilities: np.ndarray, labels: list[str]) -> dict[str, float]:
    scores = {}
    for k in (2, 3):
        if len(labels) >= k:
            scores[f"top_{k}_accuracy"] = float(top_k_accuracy_score(y_true, probabilities, k=k, labels=labels))
    return scores


def split_data(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    split_strategy: str,
    test_size: float,
):
    if split_strategy == "dog-grouped":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_index, test_index = next(splitter.split(x, y, groups))
        return (
            x.iloc[train_index],
            x.iloc[test_index],
            y.iloc[train_index],
            y.iloc[test_index],
            sorted(groups.iloc[train_index].unique()),
            sorted(groups.iloc[test_index].unique()),
        )
    if split_strategy == "random":
        stratify = y if y.value_counts().min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
        return (
            x_train,
            x_test,
            y_train,
            y_test,
            sorted(groups.loc[x_train.index].unique()),
            sorted(groups.loc[x_test.index].unique()),
        )
    raise SystemExit("--split-strategy must be either 'dog-grouped' or 'random'.")


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
    pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels).to_csv(
        report_dir / "breed_confusion_matrix.csv"
    )


def train_yamnet_classifier(
    frame: pd.DataFrame,
    model_path: Path,
    report_dir: Path,
    model_type: str,
    min_examples_per_breed: int,
    max_examples_per_breed: int | None,
    split_strategy: str,
    test_size: float,
    refit_full_data: bool,
) -> None:
    counts = frame[LABEL_COLUMN].value_counts()
    usable_breeds = counts[counts >= min_examples_per_breed].index
    frame = frame[frame[LABEL_COLUMN].isin(usable_breeds)].copy()
    if max_examples_per_breed is not None:
        frame = balanced_sample(frame, LABEL_COLUMN, max_examples_per_breed)

    feature_columns = feature_columns_for(frame)
    if not feature_columns:
        raise SystemExit("No numeric YAMNet embedding columns found.")

    x = frame[feature_columns].fillna(0.0)
    y = frame[LABEL_COLUMN].astype(str)
    groups = frame["dog_id"].astype(str)
    x_train, x_test, y_train, y_test, train_groups, test_groups = split_data(x, y, groups, split_strategy, test_size)

    model = make_model(model_type)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = list(model.classes_)
    probabilities = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None

    report_text = classification_report(y_test, predictions, zero_division=0)
    report_dict = classification_report(y_test, predictions, zero_division=0, output_dict=True)
    metrics = {
        "model_version": f"dogbridge_breed_yamnet_{model_type}_v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "feature_extractor": "yamnet",
        "yamnet_url": YAMNET_URL,
        "target_sample_rate": TARGET_SAMPLE_RATE,
        "model_type": model_type,
        "split_strategy": split_strategy,
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

    print("Breed counts used for YAMNet training:")
    print(y.value_counts().to_string())
    print()
    print(f"Split strategy: {split_strategy}")
    print(f"Model type: {model_type}")
    print(f"Train rows: {len(x_train)} | Test rows: {len(x_test)}")
    print(f"Train dogs: {len(train_groups)} | Test dogs: {len(test_groups)}")
    print()
    print(report_text)
    if "top_2_accuracy" in metrics:
        print(f"Top-2 accuracy: {metrics['top_2_accuracy']:.3f}")
    if "top_3_accuracy" in metrics:
        print(f"Top-3 accuracy: {metrics['top_3_accuracy']:.3f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.3f}")

    write_evaluation_outputs(report_dir, report_text, metrics, labels, y_test, predictions)
    print(f"Wrote YAMNet evaluation outputs to {report_dir}")

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
            "feature_extractor": "yamnet",
            "model_version": metrics["model_version"],
            "metrics": metrics,
        },
        model_path,
    )
    print(f"Saved YAMNet breed classifier to {model_path}")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Train a dog breed classifier on YAMNet audio embeddings.")
    parser.add_argument("--manifest-csv", type=Path, default=root / "backend" / "data" / "processed" / "dog_audio_manifest.csv")
    parser.add_argument("--embeddings-csv", type=Path, default=root / "backend" / "data" / "processed" / "breed_yamnet_embeddings.csv")
    parser.add_argument("--model-path", type=Path, default=root / "backend" / "data" / "models" / "breed_yamnet_classifier.joblib")
    parser.add_argument("--report-dir", type=Path, default=root / "backend" / "data" / "models" / "reports" / "breed_yamnet_classifier")
    parser.add_argument("--model-type", choices=["logistic_regression", "random_forest"], default="logistic_regression")
    parser.add_argument("--min-examples-per-breed", type=int, default=5)
    parser.add_argument("--max-examples-per-breed", type=int, default=None)
    parser.add_argument("--split-strategy", choices=["dog-grouped", "random"], default="dog-grouped")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--no-refit-full-data", action="store_true")
    args = parser.parse_args()

    if args.min_examples_per_breed < 1:
        raise SystemExit("--min-examples-per-breed must be at least 1.")
    if args.max_examples_per_breed is not None and args.max_examples_per_breed < 1:
        raise SystemExit("--max-examples-per-breed must be at least 1 when provided.")
    if not 0.0 < args.test_size < 1.0:
        raise SystemExit("--test-size must be between 0 and 1.")

    frame = build_embedding_frame(
        args.manifest_csv,
        args.embeddings_csv,
        args.max_examples_per_breed,
        args.rebuild_embeddings,
        args.checkpoint_every,
    )
    train_yamnet_classifier(
        frame,
        args.model_path,
        args.report_dir,
        args.model_type,
        args.min_examples_per_breed,
        args.max_examples_per_breed,
        args.split_strategy,
        args.test_size,
        not args.no_refit_full_data,
    )


if __name__ == "__main__":
    main()
