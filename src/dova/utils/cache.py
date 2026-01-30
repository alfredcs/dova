"""
Caching Utilities for DOVA.

Provides abstract cache interface with implementations for
in-memory (development) and Redis (production) caching.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def generate_cache_key(*args: Any, **kwargs: Any) -> str:
    """
    Generate a deterministic cache key from arguments.

    Args:
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key

    Returns:
        SHA256 hash of the arguments
    """
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = ":".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


@dataclass
class CacheEntry:
    """A cached value with metadata."""

    value: Any
    created_at: float
    ttl: float | None = None

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class Cache(ABC):
    """Abstract base class for cache implementations."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for no expiration)
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and is not expired
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all entries from the cache."""
        pass


class InMemoryCache(Cache):
    """Simple in-memory cache implementation for development/testing."""

    def __init__(self, max_size: int = 10000):
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size

    async def get(self, key: str) -> Any | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._cache[key]
            return None
        return entry.value

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        # Simple eviction: remove expired entries if at capacity
        if len(self._cache) >= self._max_size:
            self._evict_expired()

        # If still at capacity, remove oldest entries
        if len(self._cache) >= self._max_size:
            oldest_keys = sorted(
                self._cache.keys(), key=lambda k: self._cache[k].created_at
            )[: self._max_size // 10]
            for k in oldest_keys:
                del self._cache[k]

        self._cache[key] = CacheEntry(value=value, created_at=time.time(), ttl=ttl)

    async def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def exists(self, key: str) -> bool:
        entry = self._cache.get(key)
        if entry is None:
            return False
        if entry.is_expired:
            del self._cache[key]
            return False
        return True

    async def clear(self) -> None:
        self._cache.clear()

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired_keys:
            del self._cache[key]


class RedisCache(Cache):
    """Redis-backed cache implementation for production."""

    def __init__(self, url: str, prefix: str = "dova:"):
        self._url = url
        self._prefix = prefix
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy initialization of Redis client."""
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.from_url(self._url, decode_responses=True)
        return self._client

    def _make_key(self, key: str) -> str:
        """Add prefix to cache key."""
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        try:
            value = await self.client.get(self._make_key(key))
            if value is None:
                return None
            return json.loads(value)
        except Exception as e:
            logger.warning("redis_get_error", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        try:
            serialized = json.dumps(value)
            if ttl:
                await self.client.setex(self._make_key(key), int(ttl), serialized)
            else:
                await self.client.set(self._make_key(key), serialized)
        except Exception as e:
            logger.warning("redis_set_error", key=key, error=str(e))

    async def delete(self, key: str) -> bool:
        try:
            result = await self.client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.warning("redis_delete_error", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        try:
            return await self.client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.warning("redis_exists_error", key=key, error=str(e))
            return False

    async def clear(self) -> None:
        try:
            # Only clear keys with our prefix
            async for key in self.client.scan_iter(f"{self._prefix}*"):
                await self.client.delete(key)
        except Exception as e:
            logger.warning("redis_clear_error", error=str(e))

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()


def cached(
    ttl: float | None = 3600,
    key_prefix: str = "",
) -> Any:
    """
    Decorator to cache function results.

    Args:
        ttl: Time-to-live in seconds
        key_prefix: Prefix for cache keys

    Returns:
        Decorated function
    """
    # Note: This requires a cache instance to be injected
    # For now, use a simple in-memory cache
    _cache = InMemoryCache()

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = f"{key_prefix}{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            cached_value = await _cache.get(cache_key)

            if cached_value is not None:
                logger.debug("cache_hit", key=cache_key)
                return cached_value

            result = await func(*args, **kwargs)
            await _cache.set(cache_key, result, ttl=ttl)
            logger.debug("cache_miss", key=cache_key)
            return result

        return wrapper

    return decorator
