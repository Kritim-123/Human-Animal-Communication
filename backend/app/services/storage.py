from pathlib import Path
from uuid import uuid4

import librosa
from fastapi import UploadFile

from app.config import get_settings


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".flac", ".ogg"}


def save_upload(file: UploadFile, dog_id: int) -> Path:
    settings = get_settings()
    original_name = Path(file.filename or "clip.wav")
    extension = original_name.suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported audio format '{extension}'. Use one of {sorted(ALLOWED_AUDIO_EXTENSIONS)}.")

    dog_dir = settings.raw_data_dir / f"dog_{dog_id}"
    dog_dir.mkdir(parents=True, exist_ok=True)
    destination = dog_dir / f"{uuid4().hex}{extension}"

    with destination.open("wb") as output:
        while chunk := file.file.read(1024 * 1024):
            output.write(chunk)

    return destination


def get_audio_duration_seconds(path: Path) -> float | None:
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None

