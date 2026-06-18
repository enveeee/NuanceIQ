import pytest
import pytest_asyncio
import fakeredis.aioredis
from api.services.cache_service import CacheService

pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture
async def redis_client():
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield fake
    await fake.aclose()

async def test_cache_get_returns_none_when_empty(redis_client):
    svc = CacheService()
    svc._client = redis_client
    result = await svc.get("some text")
    assert result is None

async def test_cache_set_and_get(redis_client):
    svc = CacheService()
    svc._client = redis_client
    payload = {
        "label": "POSITIVE",
        "confidence": 0.98,
        "positive_score": 0.98,
        "negative_score": 0.02,
        "processing_time_ms": 40.0
    }
    await svc.set("hello world", payload)
    result = await svc.get("hello world")
    assert result is not None
    assert result["label"] == "POSITIVE"
    assert result["confidence"] == 0.98

async def test_cache_different_texts_different_keys(redis_client):
    svc = CacheService()
    svc._client = redis_client
    payload = {
        "label": "POSITIVE",
        "confidence": 0.9,
        "positive_score": 0.9,
        "negative_score": 0.1,
        "processing_time_ms": 30.0
    }
    await svc.set("text one", payload)
    result = await svc.get("text two")
    assert result is None

async def test_cache_ping_returns_true(redis_client):
    svc = CacheService()
    svc._client = redis_client
    result = await svc.ping()
    assert result is True

async def test_cache_ping_returns_false_when_no_client():
    svc = CacheService()
    svc._client = None
    result = await svc.ping()
    assert result is False