from fastapi import APIRouter

from app.api.routes import analytics, health

api_router = APIRouter(prefix="/v1")
api_router.include_router(health.router)
api_router.include_router(analytics.router)

