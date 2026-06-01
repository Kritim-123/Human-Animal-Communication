from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IntentLabel(str, Enum):
    outside_bathroom = "outside_bathroom"
    food_water = "food_water"
    play = "play"
    attention = "attention"
    stress_discomfort = "stress_discomfort"
    unknown = "unknown"


class LocationContext(str, Enum):
    door = "door"
    kitchen = "kitchen"
    couch = "couch"
    crate = "crate"
    outside = "outside"
    unknown = "unknown"


class SituationContext(str, Enum):
    before_walk = "before_walk"
    before_food = "before_food"
    owner_leaving = "owner_leaving"
    stranger_nearby = "stranger_nearby"
    toy_visible = "toy_visible"
    unknown = "unknown"


class DogCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    breed: str | None = None
    age: float | None = Field(default=None, ge=0)
    notes: str | None = None


class DogRead(DogCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AudioClipRead(BaseModel):
    id: int
    dog_id: int
    file_path: str
    duration_seconds: float | None
    recorded_at: datetime
    location_context: str
    situation_context: str
    owner_label: str
    outcome_label: str | None
    prediction_label: str | None
    prediction_confidence: float | None
    confirmed_correct: bool | None
    notes: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionRead(BaseModel):
    id: int
    clip_id: int
    predicted_label: str
    confidence: float
    top_3: list[dict[str, Any]]
    model_version: str
    created_at: datetime
    message: str | None = None


class ConfirmPrediction(BaseModel):
    confirmed_correct: bool
    corrected_label: IntentLabel | None = None
    outcome_label: IntentLabel | None = None


class StatsRead(BaseModel):
    dog_id: int
    clip_count: int
    label_distribution: dict[str, int]
    prediction_accuracy: float | None
    confirmed_count: int

