import redis.asyncio as redis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class RedisManager:
    _instance = None
    _redis_client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
        return cls._instance

    async def initialize(self):
        """Initialize Redis connection and FastAPICache"""
        try:
            self._redis_client = redis.from_url(
                settings.redis_url, 
                encoding="utf8", 
                decode_responses=True
            )
            FastAPICache.init(
                RedisBackend(self._redis_client), 
                prefix=settings.cache_prefix
            )
            logger.info("Redis connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise

    async def close(self):
        """Close Redis connection"""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Redis connection closed")

    def get_backend(self):
        """Get FastAPICache backend"""
        return FastAPICache.get_backend()

redis_manager = RedisManager()
