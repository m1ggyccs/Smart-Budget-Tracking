from typing import List, Optional

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    category: str = Field(..., description="Spending category to forecast, e.g. 'groceries'")
    periods: int = Field(4, ge=1, le=12, description="Number of future periods to forecast")
    use_user_data: bool = Field(
        True, description="Use user's transaction history if available, otherwise use pre-trained model"
    )


class ForecastDetail(BaseModel):
    model: str = Field(..., description="Forecasting technique identifier")
    data_source: str = Field(..., description="Data source driving this forecast")
    forecast: List[float]
    moving_average: Optional[dict] = None
    notes: Optional[str] = None


class ForecastResponse(BaseModel):
    category: str
    periods: int
    forecasts: List[ForecastDetail]


class CategorySpending(BaseModel):
    category: str
    amount: float


class MonthlyTrend(BaseModel):
    month: str
    total: float


class BudgetVsActual(BaseModel):
    budget_id: int
    period: str
    budget_amount: float
    actual_amount: float
    difference: float
    percentage_used: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    alerts_enabled: bool


class SpendingInsights(BaseModel):
    total_spending: float
    top_categories: List[CategorySpending]
    average_monthly: float
    trend: str  # "increasing", "decreasing", "stable"
    period_months: int

