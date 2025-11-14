"""Initial migration

Revision ID: 632ec0f37dcd
Revises: 
Create Date: 2025-11-14 15:04:57.134302

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '632ec0f37dcd'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create enums (checkfirst=True should prevent errors, but we'll be explicit)
    conn = op.get_bind()
    
    # Check and create enums only if they don't exist
    user_role_exists = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role_enum')"
    )).scalar()
    if not user_role_exists:
        user_role_enum = postgresql.ENUM('user', 'admin', name='user_role_enum', create_type=True)
        user_role_enum.create(conn, checkfirst=False)
    
    budget_period_exists = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'budget_period_enum')"
    )).scalar()
    if not budget_period_exists:
        budget_period_enum = postgresql.ENUM('weekly', 'monthly', name='budget_period_enum', create_type=True)
        budget_period_enum.create(conn, checkfirst=False)
    
    transaction_type_exists = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'transaction_type_enum')"
    )).scalar()
    if not transaction_type_exists:
        transaction_type_enum = postgresql.ENUM('expense', 'income', name='transaction_type_enum', create_type=True)
        transaction_type_enum.create(conn, checkfirst=False)
    
    # Create users table (if it doesn't exist)
    if not conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users')"
    )).scalar():
        op.create_table(
            'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('role', postgresql.ENUM('user', 'admin', name='user_role_enum', create_type=False), nullable=False, server_default='user'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
        op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    
    # Create budgets table (if it doesn't exist)
    if not conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'budgets')"
    )).scalar():
        op.create_table(
            'budgets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('period', postgresql.ENUM('weekly', 'monthly', name='budget_period_enum', create_type=False), nullable=False, server_default='monthly'),
        sa.Column('amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('start_date', sa.Date(), nullable=True),
        sa.Column('end_date', sa.Date(), nullable=True),
        sa.Column('alerts_enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )
    
    # Create transactions table (if it doesn't exist)
    if not conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'transactions')"
    )).scalar():
        op.create_table(
            'transactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('amount', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('transaction_type', postgresql.ENUM('expense', 'income', name='transaction_type_enum', create_type=False), nullable=False, server_default='expense'),
        sa.Column('occurred_at', sa.Date(), nullable=False),
        sa.Column('notes', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
        )


def downgrade() -> None:
    # Drop tables
    op.drop_table('transactions')
    op.drop_table('budgets')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
    
    # Drop enums
    transaction_type_enum = postgresql.ENUM(name='transaction_type_enum')
    transaction_type_enum.drop(op.get_bind(), checkfirst=True)
    
    budget_period_enum = postgresql.ENUM(name='budget_period_enum')
    budget_period_enum.drop(op.get_bind(), checkfirst=True)
    
    user_role_enum = postgresql.ENUM(name='user_role_enum')
    user_role_enum.drop(op.get_bind(), checkfirst=True)
