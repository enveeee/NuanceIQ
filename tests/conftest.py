import pytest
import pytest_asyncio
import fakeredis.aioredis
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch
from api.main import app
from api.services.cache_service import cache_service
from api.services.model_service import model_service

MOCK_PREDICT_RESULT = {
    "label": "POSITIVE",
    "confidence": 0.9823,
    "positive_score": 0.9823,
    "negative_score": 0.0177,
    "processing_time_ms": 42.5
}

@pytest.fixture(autouse=True)
def mock_model_service():
    with patch.object(model_service, "_loaded", True), \
         patch.object(model_service, "predict", return_value=MOCK_PREDICT_RESULT):
        yield

@pytest_asyncio.fixture
async def fake_redis():
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    cache_service._client = fake
    yield fake
    await fake.aclose()
    cache_service._client = None

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac