import json
import hashlib
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
import redis.asyncio as aioredis
from api.config import get_settings

settings = get_settings()

class CacheService:
    def __init__(self):
        self._client: aioredis.Redis | None = None

    async def connect(self):
        try:
            self._client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self._client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}. Caching disabled.")
            self._client = None

    async def disconnect(self):
        if self._client:
            await self._client.aclose()

    def _make_key(self, text: str) -> str:
        hash_val = hashlib.sha256(text.encode()).hexdigest()
        return f"nuanceiq:sentiment:{hash_val}"

    async def get(self, text: str) -> dict | None:
        if not self._client:
            return None
        try:
            key = self._make_key(text)
            value = await self._client.get(key)
            if value:
                logger.debug(f"Cache HIT for key: {key[:20]}...")
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Cache GET error: {e}")
        return None

    async def set(self, text: str, result: dict) -> None:
        if not self._client:
            return
        try:
            key = self._make_key(text)
            await self._client.setex(
                key,
                settings.cache_ttl_seconds,
                json.dumps(result)
            )
            logger.debug(f"Cache SET for key: {key[:20]}...")
        except Exception as e:
            logger.warning(f"Cache SET error: {e}")

    async def ping(self) -> bool:
        if not self._client:
            return False
        try:
            return await self._client.ping()
        except Exception:
            return False

cache_service = CacheService()