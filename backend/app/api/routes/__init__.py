from fastapi import APIRouter

from app.api.routes import admin, analytics, auth, budgets, health, transactions

api_router = APIRouter(prefix="/v1")
api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(budgets.router)
api_router.include_router(transactions.router)
api_router.include_router(analytics.router)
api_router.include_router(admin.router)

