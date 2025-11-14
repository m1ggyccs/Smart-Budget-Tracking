from datetime import date
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from app.api.routes.auth import get_current_active_user
from app.db.session import get_db
from app.models.user import User
from app.schema.analytics import (
    BudgetVsActual,
    ForecastDetail,
    ForecastRequest,
    ForecastResponse,
    MonthlyTrend,
    SpendingInsights,
)
from app.services.analytics_service import analytics_service
from app.services.model_service import forecast_service

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.post("/forecast", response_model=ForecastResponse)
def generate_forecast(
    payload: ForecastRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> ForecastResponse:
    """Generate forecast for a category using user data or pre-trained model."""
    forecasts: List[ForecastDetail] = []

    try:
        # Try to use user data if requested and available
        if payload.use_user_data:
            time_series = analytics_service.get_category_time_series(
                db, current_user.id, payload.category, months=12
            )
            if len(time_series) >= 3:
                # Simple moving average for comparison
                window = min(3, len(time_series))
                last_values = time_series[-window:]
                avg = sum(last_values) / len(last_values)
                ma_forecast = [float(avg)] * payload.periods
                forecasts.append(
                    ForecastDetail(
                        model="moving_average_user",
                        data_source="user_data",
                        forecast=ma_forecast,
                        moving_average={
                            "window": window,
                            "mean": float(avg),
                            "last_values": [float(x) for x in last_values],
                        },
                        notes="Simple moving average using recent user transactions.",
                    )
                )

            if len(time_series) >= 5:  # Need at least 5 data points for Holt-Winters
                try:
                    # Convert to pandas Series for Holt-Winters
                    series = pd.Series(time_series)

                    # Train Holt-Winters model on user data
                    hw_model = ExponentialSmoothing(
                        series,
                        trend="add",
                        seasonal=None,  # No seasonality for monthly data
                        damped_trend=True,
                    ).fit()

                    # Generate forecast
                    forecast_values = hw_model.forecast(payload.periods).tolist()

                    forecasts.append(
                        ForecastDetail(
                            model="holt_winters_user",
                            data_source="user_data",
                            forecast=[float(x) for x in forecast_values],
                            moving_average=None,
                            notes="Holt-Winters trained on your historical transactions.",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    forecasts.append(
                        ForecastDetail(
                            model="holt_winters_user",
                            data_source="user_data",
                            forecast=[],
                            notes=f"Holt-Winters training failed: {exc}",
                        )
                    )

        # Always include pre-trained model forecast for comparison
        result = forecast_service.forecast(category=payload.category, periods=payload.periods)
        forecasts.append(
            ForecastDetail(
                model="holt_winters_pretrained",
                data_source="pre_trained_model",
                forecast=[float(x) for x in result["forecast"]],
                moving_average=result.get("moving_average"),
                notes="Pre-trained Holt-Winters model using anonymized global data.",
            )
        )

        return ForecastResponse(
            category=payload.category,
            periods=payload.periods,
            forecasts=forecasts,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/category-spending", response_model=List[dict])
def get_category_spending(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
) -> List[dict]:
    """Get spending breakdown by category."""
    spending = analytics_service.get_category_spending(
        db, current_user.id, start_date=start_date, end_date=end_date
    )
    return [{"category": cat, "amount": float(amt)} for cat, amt in spending.items()]


@router.get("/monthly-trends", response_model=List[MonthlyTrend])
def get_monthly_trends(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    category: Optional[str] = Query(default=None),
    months: int = Query(default=6, ge=1, le=24),
) -> List[MonthlyTrend]:
    """Get monthly spending trends."""
    trends = analytics_service.get_monthly_trends(db, current_user.id, category=category, months=months)
    return [MonthlyTrend(**t) for t in trends]


@router.get("/budget-vs-actual", response_model=List[BudgetVsActual])
def get_budget_vs_actual(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    budget_id: Optional[int] = Query(default=None),
) -> List[BudgetVsActual]:
    """Compare budget vs actual spending."""
    results = analytics_service.get_budget_vs_actual(db, current_user.id, budget_id=budget_id)
    return [BudgetVsActual(**r) for r in results]


@router.get("/insights", response_model=SpendingInsights)
def get_spending_insights(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    months: int = Query(default=3, ge=1, le=12),
) -> SpendingInsights:
    """Get spending insights including top categories and trends."""
    insights = analytics_service.get_spending_insights(db, current_user.id, months=months)
    return SpendingInsights(**insights)

