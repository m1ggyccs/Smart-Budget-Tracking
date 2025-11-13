from functools import lru_cache
from typing import Any, Dict, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env files."""

    api_title: str = "Smart Budget Tracking API"
    api_description: str = (
        "Backend service powering budgeting, forecasting, and analytics features."
    )
    api_version: str = "0.1.0"

    debug: bool = False
    environment: str = "local"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/smart_budget"

    cors_origins: List[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        if isinstance(value, list):
            return value
        raise ValueError("Invalid CORS origins format")

    def fastapi_kwargs(self) -> Dict[str, Any]:
        return {
            "debug": self.debug,
            "title": self.api_title,
            "description": self.api_description,
            "version": self.api_version,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance to avoid re-parsing env repeatedly."""
    return Settings()

