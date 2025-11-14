#!/bin/bash
# Script to run Alembic migrations
# Usage: ./run_migrations.sh [upgrade|downgrade|revision]

set -e

cd "$(dirname "$0")"
source .venv/bin/activate

case "${1:-upgrade}" in
    upgrade)
        echo "Running database migrations..."
        alembic upgrade head
        ;;
    downgrade)
        echo "Rolling back last migration..."
        alembic downgrade -1
        ;;
    revision)
        echo "Creating new migration..."
        alembic revision --autogenerate -m "${2:-New migration}"
        ;;
    *)
        echo "Usage: $0 [upgrade|downgrade|revision] [migration_message]"
        exit 1
        ;;
esac

