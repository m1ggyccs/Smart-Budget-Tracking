from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    category: str = Field(..., description="Spending category to forecast, e.g. 'groceries'")
    periods: int = Field(4, ge=1, le=12, description="Number of future periods to forecast")


class ForecastResponse(BaseModel):
    category: str
    periods: int
    forecast: list[float]
    moving_average: dict | None = None

