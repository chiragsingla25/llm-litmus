from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.database import get_connection, init_db
from app.router import health, runs


@asynccontextmanager
async def lifespan(app: FastAPI):
    with get_connection(settings.DB_PATH) as conn:
        init_db(conn)
    yield


app = FastAPI(title="llm-litmus", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(runs.router)
