from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from sqlalchemy import Date, cast, func
from sqlalchemy.orm import Session

from app.models.user import Budget, BudgetPeriod, Transaction, TransactionType, User


class AnalyticsService:
    """Service for generating analytics and insights from user transaction data."""

    def get_category_spending(
        self,
        db: Session,
        user_id: int,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, Decimal]:
        """Get total spending per category for a user within a date range."""
        query = (
            db.query(
                Transaction.category,
                func.coalesce(func.sum(Transaction.amount), 0).label("total"),
            )
            .filter(
                Transaction.user_id == user_id,
                Transaction.transaction_type == TransactionType.EXPENSE,
            )
        )

        if start_date is not None:
            query = query.filter(Transaction.occurred_at >= start_date)
        if end_date is not None:
            query = query.filter(Transaction.occurred_at <= end_date)

        results = query.group_by(Transaction.category).all()
        return {row.category: Decimal(row.total) for row in results}

    def get_monthly_trends(
        self,
        db: Session,
        user_id: int,
        category: Optional[str] = None,
        months: int = 6,
    ) -> List[Dict[str, any]]:
        """Get monthly spending trends for a user."""
        end_date = date.today()
        start_date = end_date - timedelta(days=months * 30)

        # Use PostgreSQL's date_trunc for monthly grouping
        month_expr = func.date_trunc("month", cast(Transaction.occurred_at, Date))
        query = (
            db.query(
                month_expr.label("month"),
                func.coalesce(func.sum(Transaction.amount), 0).label("total"),
            )
            .filter(
                Transaction.user_id == user_id,
                Transaction.transaction_type == TransactionType.EXPENSE,
                Transaction.occurred_at >= start_date,
            )
        )

        if category is not None:
            query = query.filter(Transaction.category == category)

        results = query.group_by(month_expr).order_by(month_expr).all()

        return [
            {
                "month": row.month.strftime("%Y-%m") if isinstance(row.month, datetime) else str(row.month),
                "total": float(Decimal(row.total)),
            }
            for row in results
        ]

    def get_budget_vs_actual(
        self,
        db: Session,
        user_id: int,
        budget_id: Optional[int] = None,
    ) -> List[Dict[str, any]]:
        """Compare budget vs actual spending."""
        query = db.query(Budget).filter(Budget.user_id == user_id)
        if budget_id is not None:
            query = query.filter(Budget.id == budget_id)

        budgets = query.all()
        results = []

        for budget in budgets:
            # Determine date range for this budget
            if budget.start_date and budget.end_date:
                start_date = budget.start_date
                end_date = budget.end_date
            else:
                # Use current period if dates not set
                today = date.today()
                if budget.period == BudgetPeriod.MONTHLY:
                    start_date = today.replace(day=1)
                    end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                else:  # WEEKLY
                    days_since_monday = today.weekday()
                    start_date = today - timedelta(days=days_since_monday)
                    end_date = start_date + timedelta(days=6)

            # Get actual spending in this period
            actual_query = (
                db.query(func.coalesce(func.sum(Transaction.amount), 0).label("total"))
                .filter(
                    Transaction.user_id == user_id,
                    Transaction.transaction_type == TransactionType.EXPENSE,
                    Transaction.occurred_at >= start_date,
                    Transaction.occurred_at <= end_date,
                )
            )
            actual_total = Decimal(actual_query.scalar() or 0)

            budget_amount = Decimal(budget.amount)
            difference = budget_amount - actual_total
            percentage_used = (actual_total / budget_amount * 100) if budget_amount > 0 else 0

            results.append(
                {
                    "budget_id": budget.id,
                    "period": budget.period.value,
                    "budget_amount": float(budget_amount),
                    "actual_amount": float(actual_total),
                    "difference": float(difference),
                    "percentage_used": float(percentage_used),
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "alerts_enabled": budget.alerts_enabled,
                }
            )

        return results

    def get_spending_insights(
        self,
        db: Session,
        user_id: int,
        months: int = 3,
    ) -> Dict[str, any]:
        """Get spending insights including top categories, trends, and recommendations."""
        end_date = date.today()
        start_date = end_date - timedelta(days=months * 30)

        # Get category totals
        category_totals = self.get_category_spending(db, user_id, start_date, end_date)

        if not category_totals:
            return {
                "total_spending": 0.0,
                "top_categories": [],
                "average_monthly": 0.0,
                "trend": "stable",
            }

        total_spending = sum(category_totals.values())
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        top_categories = [{"category": cat, "amount": float(amt)} for cat, amt in sorted_categories[:5]]

        # Get monthly trends for trend analysis
        monthly_trends = self.get_monthly_trends(db, user_id, months=months)
        if len(monthly_trends) >= 2:
            recent_avg = sum(m["total"] for m in monthly_trends[-2:]) / 2
            older_avg = sum(m["total"] for m in monthly_trends[:-2]) / max(1, len(monthly_trends) - 2)
            if recent_avg > older_avg * 1.1:
                trend = "increasing"
            elif recent_avg < older_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        average_monthly = total_spending / months if months > 0 else 0

        return {
            "total_spending": float(total_spending),
            "top_categories": top_categories,
            "average_monthly": float(average_monthly),
            "trend": trend,
            "period_months": months,
        }

    def get_category_time_series(
        self,
        db: Session,
        user_id: int,
        category: str,
        months: int = 6,
    ) -> List[float]:
        """Get time series data for a specific category (for forecasting)."""
        end_date = date.today()
        start_date = end_date - timedelta(days=months * 30)

        month_expr = func.date_trunc("month", cast(Transaction.occurred_at, Date))
        query = (
            db.query(
                month_expr.label("month"),
                func.coalesce(func.sum(Transaction.amount), 0).label("total"),
            )
            .filter(
                Transaction.user_id == user_id,
                Transaction.category == category,
                Transaction.transaction_type == TransactionType.EXPENSE,
                Transaction.occurred_at >= start_date,
            )
            .group_by(month_expr)
            .order_by(month_expr)
        )

        results = query.all()
        return [float(Decimal(row.total)) for row in results]


analytics_service = AnalyticsService()

