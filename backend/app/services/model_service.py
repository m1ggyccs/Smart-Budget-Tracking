from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import get_settings


class ForecastModelService:
    """Loads pre-trained forecasting artifacts and exposes helper prediction utilities."""

    def __init__(self, model_dir: Path | None = None) -> None:
        settings = get_settings()
        self.model_dir = self._resolve_model_dir(model_dir, settings.models_directory)
        self._holt_winters_models: Dict[str, Any] = {}
        self._moving_average_configs: Dict[str, Dict[str, Any]] = {}

    def _resolve_model_dir(self, override: Path | None, configured_path: str) -> Path:
        if override is not None:
            return override.resolve()

        candidate = Path(configured_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate

        search_roots = [
            Path(__file__).resolve().parents[2],  # /app inside container
            Path(__file__).resolve().parents[3],  # project root during local dev
            Path.cwd(),
        ]
        for root in search_roots:
            potential = (root / candidate).resolve()
            if potential.exists():
                return potential

        # Default to container-style path even if it doesn't exist yet; forecast() will validate.
        return (Path(__file__).resolve().parents[2] / candidate).resolve()

    def load_models(self) -> None:
        """Eagerly load serialized models/configs into memory."""
        self._holt_winters_models = self._load_pickled_models("holtwinters")
        self._moving_average_configs = self._load_json_configs("moving_average")

    def _load_pickled_models(self, prefix: str) -> Dict[str, Any]:
        models: Dict[str, Any] = {}
        for artifact in self.model_dir.glob(f"{prefix}_*.pkl"):
            with artifact.open("rb") as f:
                models[artifact.stem.replace(f"{prefix}_", "")] = pickle.load(f)
        return models

    def _load_json_configs(self, prefix: str) -> Dict[str, Dict[str, Any]]:
        configs: Dict[str, Dict[str, Any]] = {}
        for artifact in self.model_dir.glob(f"{prefix}_*.json"):
            with artifact.open("r", encoding="utf-8") as f:
                configs[artifact.stem.replace(f"{prefix}_", "")] = json.load(f)
        return configs

    def supported_categories(self) -> List[str]:
        categories = set(self._holt_winters_models.keys()) | set(self._moving_average_configs.keys())
        return sorted(categories)

    def forecast(self, category: str, periods: int = 4) -> Dict[str, Any]:
        if not self._holt_winters_models:
            self.load_models()

        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        if category not in self._holt_winters_models:
            raise ValueError(f"Category '{category}' is not available. Available: {self.supported_categories()}")

        model = self._holt_winters_models[category]
        forecast_values = model.forecast(periods)

        response: Dict[str, Any] = {
            "category": category,
            "periods": periods,
            "forecast": list(map(float, forecast_values)),
        }

        if category in self._moving_average_configs:
            response["moving_average"] = self._moving_average_configs[category]

        return response


forecast_service = ForecastModelService()

