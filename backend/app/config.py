from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DogBridge"
    database_url: str = "sqlite:///./dogbridge.db"
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    model_path: Path = Path("data/models/dogbridge_baseline.joblib")
    confidence_threshold: float = 0.55

    model_config = SettingsConfigDict(env_prefix="DOGBRIDGE_")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

