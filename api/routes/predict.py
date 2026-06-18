import io
import time
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from api.schemas.sentiment import (
    PredictRequest, PredictResponse, BatchPredictResponse
)
from api.services.model_service import model_service
from api.services.cache_service import cache_service
from api.config import get_settings

settings = get_settings()
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(tags=["Prediction"])

@router.post("/predict", response_model=PredictResponse)
@limiter.limit(settings.rate_limit)
async def predict_sentiment(
    request: Request,
    body: PredictRequest
):
    cached = await cache_service.get(body.text)
    if cached:
        return PredictResponse(text=body.text, cached=True, **cached)

    try:
        result = model_service.predict(body.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    await cache_service.set(body.text, result)

    return PredictResponse(
        text=body.text,
        cached=False,
        **result
    )

@router.post("/predict/batch", response_model=BatchPredictResponse)
@limiter.limit("10/minute")
async def predict_batch(
    request: Request,
    file: UploadFile = File(..., description="CSV file with a 'text' column")
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted"
        )

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse CSV file")

    if "text" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain a 'text' column"
        )

    df = df.dropna(subset=["text"])
    if len(df) > 500:
        raise HTTPException(
            status_code=400,
            detail="Batch limit is 500 rows per request"
        )

    results = []
    for text in df["text"].tolist():
        text = str(text).strip()
        cached = await cache_service.get(text)
        if cached:
            results.append(
                PredictResponse(text=text, cached=True, **cached)
            )
            continue
        try:
            result = model_service.predict(text)
            await cache_service.set(text, result)
            results.append(
                PredictResponse(text=text, cached=False, **result)
            )
        except Exception:
            continue

    return BatchPredictResponse(total=len(results), results=results)