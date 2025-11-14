from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import api_router
from app.core.config import get_settings
from app.db import base  # noqa: F401
from app.db.session import engine

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Ensure models are imported for Alembic autogenerate support.
    # For production, use migrations: `alembic upgrade head`
    # For development, you can enable auto-create by setting AUTO_CREATE_TABLES=true
    import os
    if os.getenv("AUTO_CREATE_TABLES", "false").lower() == "true":
        base.Base.metadata.create_all(bind=engine)
    yield


def create_application() -> FastAPI:
    app = FastAPI(lifespan=lifespan, **settings.fastapi_kwargs())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    return app


app = create_application()

