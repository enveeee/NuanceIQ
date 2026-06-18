from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum

class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      description="Text to analyze")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only")
        return v.strip()

class PredictResponse(BaseModel):
    text: str
    label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0,
                               description="Model confidence score (0–1)")
    positive_score: float = Field(..., ge=0.0, le=1.0)
    negative_score: float = Field(..., ge=0.0, le=1.0)
    cached: bool = Field(default=False,
                         description="True if result served from Redis cache")
    processing_time_ms: Optional[float] = None

class BatchPredictResponse(BaseModel):
    total: int
    results: list[PredictResponse]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    version: str = "1.0.0"

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    model: str
    dataset: str