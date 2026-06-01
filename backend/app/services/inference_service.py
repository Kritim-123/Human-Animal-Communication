import json
from pathlib import Path
from typing import Any

import joblib

from app.config import get_settings
from scripts.extract_features import extract_feature_dict


STRESS_MESSAGE = "Monitor your dog. If this is unusual or persistent, contact a veterinarian."


class ModelNotReadyError(RuntimeError):
    pass


def predict_clip(file_path: str | Path) -> dict[str, Any]:
    settings = get_settings()
    if not settings.model_path.exists():
        raise ModelNotReadyError(
            f"No trained model found at {settings.model_path}. Collect labeled real audio clips, then run train_baseline.py first."
        )

    bundle = joblib.load(settings.model_path)
    model = bundle["model"]
    labels = list(bundle["labels"])
    feature_names = list(bundle["feature_names"])
    model_version = bundle.get("model_version", "dogbridge_baseline")

    features = extract_feature_dict(Path(file_path))
    feature_vector = [[features[name] for name in feature_names]]

    probabilities = model.predict_proba(feature_vector)[0]
    ranked = sorted(zip(labels, probabilities, strict=True), key=lambda item: item[1], reverse=True)
    best_label, best_confidence = ranked[0]

    if best_confidence < settings.confidence_threshold:
        predicted_label = "unknown"
        confidence = float(best_confidence)
    else:
        predicted_label = str(best_label)
        confidence = float(best_confidence)

    top_3 = [{"label": str(label), "confidence": float(prob)} for label, prob in ranked[:3]]
    if "unknown" not in {item["label"] for item in top_3}:
        top_3.append({"label": "unknown", "confidence": 0.0})
        top_3 = top_3[:3]

    message = STRESS_MESSAGE if predicted_label == "stress_discomfort" else None

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top_3": top_3,
        "top_3_json": json.dumps(top_3),
        "model_version": model_version,
        "message": message,
    }

