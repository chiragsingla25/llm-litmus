from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.router import health

app = FastAPI(title="llm-litmus")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
