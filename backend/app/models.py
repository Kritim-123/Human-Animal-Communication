from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Dog(Base):
    __tablename__ = "dogs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    breed: Mapped[str | None] = mapped_column(String(120), nullable=True)
    age: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    clips: Mapped[list["AudioClip"]] = relationship(back_populates="dog", cascade="all, delete-orphan")


class AudioClip(Base):
    __tablename__ = "audio_clips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dog_id: Mapped[int] = mapped_column(ForeignKey("dogs.id"), nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    location_context: Mapped[str] = mapped_column(String(80), default="unknown")
    situation_context: Mapped[str] = mapped_column(String(80), default="unknown")
    owner_label: Mapped[str] = mapped_column(String(80), default="unknown")
    outcome_label: Mapped[str | None] = mapped_column(String(80), nullable=True)
    prediction_label: Mapped[str | None] = mapped_column(String(80), nullable=True)
    prediction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confirmed_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    dog: Mapped[Dog] = relationship(back_populates="clips")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="clip", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    clip_id: Mapped[int] = mapped_column(ForeignKey("audio_clips.id"), nullable=False, index=True)
    predicted_label: Mapped[str] = mapped_column(String(80), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top_3_json: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(120), default="dogbridge_baseline")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    clip: Mapped[AudioClip] = relationship(back_populates="predictions")

