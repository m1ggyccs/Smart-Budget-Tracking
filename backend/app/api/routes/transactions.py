from datetime import date
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_active_user
from app.db.session import get_db
from app.models.user import Transaction, TransactionType, User
from app.schema.transaction import (
    TransactionCreate,
    TransactionRead,
    TransactionSummary,
    TransactionUpdate,
)

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post("", response_model=TransactionRead, status_code=status.HTTP_201_CREATED)
def create_transaction(
    payload: TransactionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> TransactionRead:
    transaction = Transaction(
        user_id=current_user.id,
        amount=payload.amount,
        category=payload.category,
        transaction_type=payload.transaction_type,
        occurred_at=payload.occurred_at,
        notes=payload.notes,
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return TransactionRead.model_validate(transaction)


@router.get("", response_model=List[TransactionRead])
def list_transactions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    transaction_type: Optional[TransactionType] = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> List[TransactionRead]:
    query = db.query(Transaction).filter(Transaction.user_id == current_user.id)
    if start_date is not None:
        query = query.filter(Transaction.occurred_at >= start_date)
    if end_date is not None:
        query = query.filter(Transaction.occurred_at <= end_date)
    if transaction_type is not None:
        query = query.filter(Transaction.transaction_type == transaction_type)

    transactions = (
        query.order_by(Transaction.occurred_at.desc()).offset(skip).limit(limit).all()
    )
    return [TransactionRead.model_validate(t) for t in transactions]


@router.get("/summary", response_model=TransactionSummary)
def transaction_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
) -> TransactionSummary:
    query = db.query(
        Transaction.transaction_type,
        func.coalesce(func.sum(Transaction.amount), 0).label("total"),
    ).filter(Transaction.user_id == current_user.id)

    if start_date is not None:
        query = query.filter(Transaction.occurred_at >= start_date)
    if end_date is not None:
        query = query.filter(Transaction.occurred_at <= end_date)

    query = query.group_by(Transaction.transaction_type)

    totals = {
        TransactionType.EXPENSE: Decimal("0"),
        TransactionType.INCOME: Decimal("0"),
    }
    for row in query.all():
        totals[row.transaction_type] = Decimal(row.total)

    expense_total: Decimal = totals[TransactionType.EXPENSE]
    income_total: Decimal = totals[TransactionType.INCOME]
    net: Decimal = income_total - expense_total

    return TransactionSummary(
        total_expense=expense_total,
        total_income=income_total,
        net=net,
    )


@router.get("/{transaction_id}", response_model=TransactionRead)
def get_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> TransactionRead:
    transaction = (
        db.query(Transaction)
        .filter(Transaction.id == transaction_id, Transaction.user_id == current_user.id)
        .first()
    )
    if transaction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    return TransactionRead.model_validate(transaction)


@router.patch("/{transaction_id}", response_model=TransactionRead)
def update_transaction(
    transaction_id: int,
    payload: TransactionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> TransactionRead:
    transaction = (
        db.query(Transaction)
        .filter(Transaction.id == transaction_id, Transaction.user_id == current_user.id)
        .first()
    )
    if transaction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(transaction, key, value)

    db.commit()
    db.refresh(transaction)
    return TransactionRead.model_validate(transaction)


@router.delete("/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> None:
    transaction = (
        db.query(Transaction)
        .filter(Transaction.id == transaction_id, Transaction.user_id == current_user.id)
        .first()
    )
    if transaction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    db.delete(transaction)
    db.commit()

