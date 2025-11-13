from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from app.models.user import BudgetPeriod


class BudgetBase(BaseModel):
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    amount: Decimal = Field(gt=0)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    alerts_enabled: bool = True


class BudgetCreate(BudgetBase):
    pass


class BudgetUpdate(BaseModel):
    period: Optional[BudgetPeriod] = None
    amount: Optional[Decimal] = Field(default=None, gt=0)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    alerts_enabled: Optional[bool] = None


class BudgetRead(BudgetBase):
    id: int

    class Config:
        from_attributes = True

