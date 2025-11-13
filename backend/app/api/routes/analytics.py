from fastapi import APIRouter, HTTPException

from app.schema.analytics import ForecastRequest, ForecastResponse
from app.services.model_service import forecast_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.post("/forecast", response_model=ForecastResponse)
def generate_forecast(payload: ForecastRequest) -> ForecastResponse:
    try:
        result = forecast_service.forecast(category=payload.category, periods=payload.periods)
        return ForecastResponse(**result)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

