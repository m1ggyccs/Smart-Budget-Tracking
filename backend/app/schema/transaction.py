from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from app.models.user import TransactionType


class TransactionBase(BaseModel):
    amount: Decimal = Field(gt=0)
    category: str = Field(min_length=1, max_length=100)
    transaction_type: TransactionType = TransactionType.EXPENSE
    occurred_at: date
    notes: Optional[str] = Field(default=None, max_length=500)


class TransactionCreate(TransactionBase):
    pass


class TransactionUpdate(BaseModel):
    amount: Optional[Decimal] = Field(default=None, gt=0)
    category: Optional[str] = Field(default=None, min_length=1, max_length=100)
    transaction_type: Optional[TransactionType] = None
    occurred_at: Optional[date] = None
    notes: Optional[str] = Field(default=None, max_length=500)


class TransactionRead(TransactionBase):
    id: int

    class Config:
        from_attributes = True


class TransactionSummary(BaseModel):
    total_expense: Decimal = Field(default=0)
    total_income: Decimal = Field(default=0)
    net: Decimal = Field(default=0)

