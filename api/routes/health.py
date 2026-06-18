import json
from pathlib import Path
from fastapi import APIRouter
from api.schemas.sentiment import HealthResponse, MetricsResponse
from api.services.model_service import model_service
from api.services.cache_service import cache_service
from api.config import get_settings

settings = get_settings()
router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    redis_ok = await cache_service.ping()
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        redis_connected=redis_ok
    )

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    metrics_path = Path(settings.model_path) / "metrics.json"
    if not metrics_path.exists():
        return MetricsResponse(
            accuracy=0.0, precision=0.0,
            recall=0.0, f1=0.0,
            model="distilbert-base-uncased",
            dataset="imdb"
        )
    with open(metrics_path) as f:
        data = json.load(f)
    return MetricsResponse(**data)