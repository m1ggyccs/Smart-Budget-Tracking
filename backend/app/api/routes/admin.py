from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.routes.auth import get_current_active_user
from app.db.session import get_db
from app.models.user import Budget, Transaction, User, UserRole
from app.schema.budget import BudgetRead, BudgetUpdate
from app.schema.transaction import TransactionRead, TransactionUpdate
from app.schema.user import UserRead, UserUpdate

router = APIRouter(prefix="/admin", tags=["admin"])


def get_current_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """Dependency to ensure the current user is an admin."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin access required.",
        )
    return current_user


# User Management Endpoints
@router.get("/users", response_model=List[UserRead])
def list_all_users(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    is_active: Optional[bool] = Query(default=None),
) -> List[UserRead]:
    """List all users. Admin only."""
    query = db.query(User)
    if is_active is not None:
        query = query.filter(User.is_active == is_active)
    users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()
    return [UserRead.model_validate(u) for u in users]


@router.get("/users/{user_id}", response_model=UserRead)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> UserRead:
    """Get user details by ID. Admin only."""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserRead.model_validate(user)


@router.patch("/users/{user_id}", response_model=UserRead)
def update_user(
    user_id: int,
    payload: UserUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> UserRead:
    """Update user information. Admin only."""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)

    db.commit()
    db.refresh(user)
    return UserRead.model_validate(user)


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> None:
    """Delete a user and all their data. Admin only."""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    db.delete(user)
    db.commit()


# Budget Management Endpoints (Admin can view/edit all budgets)
@router.get("/budgets", response_model=List[BudgetRead])
def list_all_budgets(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    user_id: Optional[int] = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> List[BudgetRead]:
    """List all budgets across all users. Admin only."""
    query = db.query(Budget)
    if user_id is not None:
        query = query.filter(Budget.user_id == user_id)
    budgets = query.order_by(Budget.created_at.desc()).offset(skip).limit(limit).all()
    return [BudgetRead.model_validate(b) for b in budgets]


@router.get("/budgets/{budget_id}", response_model=BudgetRead)
def get_budget(
    budget_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> BudgetRead:
    """Get budget by ID. Admin only."""
    budget = db.query(Budget).filter(Budget.id == budget_id).first()
    if budget is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Budget not found")
    return BudgetRead.model_validate(budget)


@router.patch("/budgets/{budget_id}", response_model=BudgetRead)
def update_budget(
    budget_id: int,
    payload: BudgetUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> BudgetRead:
    """Update any user's budget. Admin only."""
    budget = db.query(Budget).filter(Budget.id == budget_id).first()
    if budget is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Budget not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(budget, key, value)

    db.commit()
    db.refresh(budget)
    return BudgetRead.model_validate(budget)


@router.delete("/budgets/{budget_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_budget(
    budget_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> None:
    """Delete any user's budget. Admin only."""
    budget = db.query(Budget).filter(Budget.id == budget_id).first()
    if budget is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Budget not found")
    db.delete(budget)
    db.commit()


# Transaction Management Endpoints (Admin can view/edit all transactions)
@router.get("/transactions", response_model=List[TransactionRead])
def list_all_transactions(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
    user_id: Optional[int] = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> List[TransactionRead]:
    """List all transactions across all users. Admin only."""
    query = db.query(Transaction)
    if user_id is not None:
        query = query.filter(Transaction.user_id == user_id)
    transactions = query.order_by(Transaction.occurred_at.desc()).offset(skip).limit(limit).all()
    return [TransactionRead.model_validate(t) for t in transactions]


@router.get("/transactions/{transaction_id}", response_model=TransactionRead)
def get_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> TransactionRead:
    """Get transaction by ID. Admin only."""
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if transaction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    return TransactionRead.model_validate(transaction)


@router.patch("/transactions/{transaction_id}", response_model=TransactionRead)
def update_transaction(
    transaction_id: int,
    payload: TransactionUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> TransactionRead:
    """Update any user's transaction. Admin only."""
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if transaction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(transaction, key, value)

    db.commit()
    db.refresh(transaction)
    return TransactionRead.model_validate(transaction)


@router.delete("/transactions/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_transaction(
    transaction_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
) -> None:
    """Delete any user's transaction. Admin only."""
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if transaction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transaction not found")
    db.delete(transaction)
    db.commit()

