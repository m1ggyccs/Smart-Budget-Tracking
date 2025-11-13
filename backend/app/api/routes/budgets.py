from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_active_user
from app.db.session import get_db
from app.models.user import Budget, BudgetPeriod, User
from app.schema.budget import BudgetCreate, BudgetRead, BudgetUpdate

router = APIRouter(prefix="/budgets", tags=["budgets"])


@router.post("", response_model=BudgetRead, status_code=status.HTTP_201_CREATED)
def create_budget(
    payload: BudgetCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> BudgetRead:
    budget = Budget(
        user_id=current_user.id,
        period=payload.period,
        amount=payload.amount,
        start_date=payload.start_date,
        end_date=payload.end_date,
        alerts_enabled=payload.alerts_enabled,
    )
    db.add(budget)
    db.commit()
    db.refresh(budget)
    return BudgetRead.model_validate(budget)


@router.get("", response_model=List[BudgetRead])
def list_budgets(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    period: Optional[BudgetPeriod] = Query(default=None),
) -> List[BudgetRead]:
    query = db.query(Budget).filter(Budget.user_id == current_user.id)
    if period is not None:
        query = query.filter(Budget.period == period)
    budgets = query.order_by(Budget.start_date.desc().nullslast()).all()
    return [BudgetRead.model_validate(b) for b in budgets]


@router.get("/{budget_id}", response_model=BudgetRead)
def get_budget(
    budget_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> BudgetRead:
    budget = (
        db.query(Budget)
        .filter(Budget.id == budget_id, Budget.user_id == current_user.id)
        .first()
    )
    if budget is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Budget not found")
    return BudgetRead.model_validate(budget)


@router.patch("/{budget_id}", response_model=BudgetRead)
def update_budget(
    budget_id: int,
    payload: BudgetUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> BudgetRead:
    budget = (
        db.query(Budget)
        .filter(Budget.id == budget_id, Budget.user_id == current_user.id)
        .first()
    )
    if budget is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Budget not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(budget, key, value)

    db.commit()
    db.refresh(budget)
    return BudgetRead.model_validate(budget)


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_budget(
    budget_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> None:
    budget = (
        db.query(Budget)
        .filter(Budget.id == budget_id, Budget.user_id == current_user.id)
        .first()
    )
    if budget is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Budget not found")
    db.delete(budget)
    db.commit()

