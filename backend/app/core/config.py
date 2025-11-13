from functools import lru_cache
from typing import Any, Dict, List, Union

from pydantic import Field, field_validator
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

    cors_origins: Union[List[str], str] = Field(default_factory=lambda: ["http://localhost:5173"])
    models_directory: str = Field(default="models")
    secret_key: str = Field(default="change-me")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value: Any) -> List[str]:
        if value is None:
            return ["http://localhost:5173"]
        if isinstance(value, str):
            if not value.strip():
                return []
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

