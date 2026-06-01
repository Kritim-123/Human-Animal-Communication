from collections import Counter

from sqlalchemy import func
from sqlalchemy.orm import Session

from app import models, schemas


def create_dog(db: Session, dog: schemas.DogCreate) -> models.Dog:
    db_dog = models.Dog(**dog.model_dump())
    db.add(db_dog)
    db.commit()
    db.refresh(db_dog)
    return db_dog


def list_dogs(db: Session) -> list[models.Dog]:
    return db.query(models.Dog).order_by(models.Dog.created_at.desc()).all()


def get_dog(db: Session, dog_id: int) -> models.Dog | None:
    return db.get(models.Dog, dog_id)


def create_clip(
    db: Session,
    *,
    dog_id: int,
    file_path: str,
    duration_seconds: float | None,
    location_context: str,
    situation_context: str,
    owner_label: str,
    outcome_label: str | None,
    notes: str | None,
) -> models.AudioClip:
    clip = models.AudioClip(
        dog_id=dog_id,
        file_path=file_path,
        duration_seconds=duration_seconds,
        location_context=location_context,
        situation_context=situation_context,
        owner_label=owner_label,
        outcome_label=outcome_label,
        notes=notes,
    )
    db.add(clip)
    db.commit()
    db.refresh(clip)
    return clip


def list_clips(db: Session) -> list[models.AudioClip]:
    return db.query(models.AudioClip).order_by(models.AudioClip.created_at.desc()).all()


def get_clip(db: Session, clip_id: int) -> models.AudioClip | None:
    return db.get(models.AudioClip, clip_id)


def create_prediction(
    db: Session,
    *,
    clip: models.AudioClip,
    predicted_label: str,
    confidence: float,
    top_3_json: str,
    model_version: str,
) -> models.Prediction:
    prediction = models.Prediction(
        clip_id=clip.id,
        predicted_label=predicted_label,
        confidence=confidence,
        top_3_json=top_3_json,
        model_version=model_version,
    )
    clip.prediction_label = predicted_label
    clip.prediction_confidence = confidence
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    db.refresh(clip)
    return prediction


def confirm_prediction(
    db: Session,
    *,
    clip: models.AudioClip,
    confirmation: schemas.ConfirmPrediction,
) -> models.AudioClip:
    clip.confirmed_correct = confirmation.confirmed_correct
    if confirmation.corrected_label:
        clip.owner_label = confirmation.corrected_label.value
    if confirmation.outcome_label:
        clip.outcome_label = confirmation.outcome_label.value
    db.commit()
    db.refresh(clip)
    return clip


def get_stats(db: Session, dog_id: int) -> schemas.StatsRead:
    clips = db.query(models.AudioClip).filter(models.AudioClip.dog_id == dog_id).all()
    distribution = Counter(clip.owner_label for clip in clips)
    confirmed = [clip for clip in clips if clip.confirmed_correct is not None]
    accuracy = None
    if confirmed:
        accuracy = sum(1 for clip in confirmed if clip.confirmed_correct) / len(confirmed)
    return schemas.StatsRead(
        dog_id=dog_id,
        clip_count=len(clips),
        label_distribution=dict(distribution),
        prediction_accuracy=accuracy,
        confirmed_count=len(confirmed),
    )

