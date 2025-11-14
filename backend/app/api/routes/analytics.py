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
    """Generate forecast for monthly budget adherence using user's total spending history."""
    from app.models.user import Budget, BudgetPeriod

    # Get the budget to forecast against
    if payload.budget_id:
        budget = db.query(Budget).filter(
            Budget.id == payload.budget_id,
            Budget.user_id == current_user.id
        ).first()
        if not budget:
            raise HTTPException(status_code=404, detail="Budget not found")
    else:
        # Get active monthly budget
        budget = db.query(Budget).filter(
            Budget.user_id == current_user.id,
            Budget.period == BudgetPeriod.MONTHLY
        ).order_by(Budget.created_at.desc()).first()
        if not budget:
            raise HTTPException(status_code=404, detail="No monthly budget found. Please create a budget first.")

    budget_amount = float(budget.amount)
    forecasts: List[ForecastDetail] = []

    try:
        # Get monthly total spending history (all expenses, not category-specific)
        time_series = analytics_service.get_monthly_total_spending(
            db, current_user.id, months=12
        )

        # Calculate accuracy metrics helper
        def calculate_accuracy(actual: List[float], predicted: List[float]) -> dict:
            """Calculate accuracy metrics for model evaluation."""
            if len(actual) < 2 or len(predicted) == 0:
                return {"accuracy": "N/A", "mae": "N/A", "confidence": "Low"}
            
            # Use last N actual values to compare with predictions
            # For simplicity, compare recent trend with forecast trend
            if len(actual) >= len(predicted):
                recent_actual = actual[-len(predicted):]
                errors = [abs(a - p) for a, p in zip(recent_actual, predicted[:len(recent_actual)])]
                mae = sum(errors) / len(errors) if errors else 0
                avg_actual = sum(recent_actual) / len(recent_actual)
                accuracy_pct = max(0, 100 - (mae / avg_actual * 100)) if avg_actual > 0 else 0
                confidence = "High" if accuracy_pct > 80 else "Medium" if accuracy_pct > 60 else "Low"
                return {
                    "accuracy": f"{accuracy_pct:.1f}%",
                    "mae": f"${mae:.2f}",
                    "confidence": confidence
                }
            return {"accuracy": "N/A", "mae": "N/A", "confidence": "Low"}

        # Model 1: Moving Average (always generate if we have any data)
        if len(time_series) >= 1:
            if len(time_series) >= 3:
                window = min(3, len(time_series))
                last_values = time_series[-window:]
                avg = sum(last_values) / len(last_values)
            else:
                # Use all available data
                avg = sum(time_series) / len(time_series)
                last_values = time_series
            
            ma_forecast = [float(avg)] * payload.periods
            accuracy = calculate_accuracy(time_series, ma_forecast)
            
            forecasts.append(
                ForecastDetail(
                    model="moving_average_user",
                    data_source="user_data",
                    forecast=ma_forecast,
                    moving_average={
                        "window": min(3, len(time_series)),
                        "mean": float(avg),
                        "last_values": [float(x) for x in last_values[-3:]] if len(last_values) >= 3 else [float(x) for x in last_values],
                        "budget_adherence": [f"{((f / budget_amount) * 100):.1f}%" for f in ma_forecast],
                        "accuracy": accuracy,
                    },
                    notes=f"Simple moving average using last {min(3, len(time_series))} months. Forecasted vs ${budget_amount:.2f} monthly budget.",
                )
            )
        else:
            # No data - use budget as baseline
            forecasts.append(
                ForecastDetail(
                    model="moving_average_user",
                    data_source="budget",
                    forecast=[budget_amount] * payload.periods,
                    moving_average={
                        "accuracy": {"accuracy": "N/A", "mae": "N/A", "confidence": "Low"},
                    },
                    notes=f"No spending history. Using budget target of ${budget_amount:.2f} per month.",
                )
            )

        # Model 2: Holt-Winters (try if enough data, otherwise use average)
        if len(time_series) >= 5:
            try:
                series = pd.Series(time_series)
                hw_model = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal=None,
                    damped_trend=True,
                ).fit()
                forecast_values = hw_model.forecast(payload.periods).tolist()
                hw_forecast = [float(x) for x in forecast_values]
                accuracy = calculate_accuracy(time_series, hw_forecast)
                
                forecasts.append(
                    ForecastDetail(
                        model="holt_winters_user",
                        data_source="user_data",
                        forecast=hw_forecast,
                        moving_average={
                            "budget_adherence": [f"{((f / budget_amount) * 100):.1f}%" for f in hw_forecast],
                            "accuracy": accuracy,
                        },
                        notes=f"Holt-Winters exponential smoothing trained on {len(time_series)} months of data. Forecasted vs ${budget_amount:.2f} monthly budget.",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # Fallback to average if Holt-Winters fails
                avg_spending = sum(time_series) / len(time_series) if time_series else budget_amount
                fallback_forecast = [float(avg_spending)] * payload.periods
                accuracy = calculate_accuracy(time_series, fallback_forecast) if time_series else {"accuracy": "N/A", "mae": "N/A", "confidence": "Low"}
                
                forecasts.append(
                    ForecastDetail(
                        model="holt_winters_user",
                        data_source="user_data",
                        forecast=fallback_forecast,
                        moving_average={
                            "budget_adherence": [f"{((f / budget_amount) * 100):.1f}%" for f in fallback_forecast],
                            "accuracy": accuracy,
                        },
                        notes=f"Holt-Winters unavailable. Using average spending of ${avg_spending:.2f} vs ${budget_amount:.2f} budget.",
                    )
                )
        elif len(time_series) > 0:
            # Not enough data for Holt-Winters, use average
            avg_spending = sum(time_series) / len(time_series)
            fallback_forecast = [float(avg_spending)] * payload.periods
            accuracy = calculate_accuracy(time_series, fallback_forecast)
            
            forecasts.append(
                ForecastDetail(
                    model="holt_winters_user",
                    data_source="user_data",
                    forecast=fallback_forecast,
                    moving_average={
                        "budget_adherence": [f"{((f / budget_amount) * 100):.1f}%" for f in fallback_forecast],
                        "accuracy": accuracy,
                    },
                    notes=f"Insufficient data for Holt-Winters (need 5+ months, have {len(time_series)}). Using average spending of ${avg_spending:.2f} vs ${budget_amount:.2f} budget.",
                )
            )
        else:
            # No data at all
            forecasts.append(
                ForecastDetail(
                    model="holt_winters_user",
                    data_source="budget",
                    forecast=[budget_amount] * payload.periods,
                    moving_average={
                        "accuracy": {"accuracy": "N/A", "mae": "N/A", "confidence": "Low"},
                    },
                    notes=f"No spending history. Using budget target of ${budget_amount:.2f} per month.",
                )
            )

        # Model 3: Average/Historical Baseline
        if len(time_series) > 0:
            avg_spending = sum(time_series) / len(time_series)
            baseline_forecast = [float(avg_spending)] * payload.periods
            accuracy = calculate_accuracy(time_series, baseline_forecast)
            
            forecasts.append(
                ForecastDetail(
                    model="historical_average",
                    data_source="user_data",
                    forecast=baseline_forecast,
                    moving_average={
                        "budget_adherence": [f"{((f / budget_amount) * 100):.1f}%" for f in baseline_forecast],
                        "accuracy": accuracy,
                    },
                    notes=f"Historical average based on {len(time_series)} month(s) of data. Average: ${avg_spending:.2f} vs ${budget_amount:.2f} budget.",
                )
            )
        else:
            # No data - use budget as baseline
            forecasts.append(
                ForecastDetail(
                    model="historical_average",
                    data_source="budget",
                    forecast=[budget_amount] * payload.periods,
                    moving_average={
                        "accuracy": {"accuracy": "N/A", "mae": "N/A", "confidence": "Low"},
                    },
                    notes=f"No spending history. Using budget target of ${budget_amount:.2f} per month as baseline.",
                )
            )

        return ForecastResponse(
            budget_id=budget.id,
            budget_amount=budget_amount,
            periods=payload.periods,
            forecasts=forecasts,
        )
    except HTTPException:
        raise
    except Exception as exc:
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

