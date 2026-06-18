"""
NuanceIQ — FastAPI Application Entry Point
Run: uvicorn api.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger
import sys

from api.config import get_settings
from api.middleware.logging import RequestLoggingMiddleware
from api.routes import predict, health
from api.services.model_service import model_service
from api.services.cache_service import cache_service

settings = get_settings()

# ── Logging ───────────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level}</level> | {message}",
    level=settings.log_level
)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NuanceIQ starting up...")
    await cache_service.connect()
    model_service.load()
    logger.info("NuanceIQ ready.")
    yield
    logger.info("NuanceIQ shutting down...")
    await cache_service.disconnect()

# ── App ───────────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="NuanceIQ",
    description="Real-Time Sentiment Intelligence API powered by DistilBERT",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(health.router)
app.include_router(predict.router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "NuanceIQ",
        "description": "Real-Time Sentiment Intelligence API",
        "docs": "/docs",
        "health": "/health"
    }