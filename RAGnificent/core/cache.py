"""
Cache module for HTTP requests to avoid repeated network calls.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("request_cache")


class RequestCache:
    """Simple cache for HTTP requests to avoid repeated network calls."""

    def __init__(self, cache_dir: str = ".request_cache", max_age: int = 3600,
                 max_memory_items: int = 100, max_memory_size_mb: int = 50):
        """
        Initialize the request cache.

        Args:
            cache_dir: Directory to store cached responses
            max_age: Maximum age of cached responses in seconds (default: 1 hour)
            max_memory_items: Maximum number of items to keep in memory cache
            max_memory_size_mb: Maximum size of memory cache in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age
        self.max_memory_items = max_memory_items
        self.max_memory_size_mb = max_memory_size_mb
        self.memory_cache: Dict[
            str, Tuple[str, float]
        ] = {}  # url -> (content, timestamp)
        self.current_memory_usage = 0  # Approximate memory usage in bytes

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key from a URL."""
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, url: str) -> Path:
        """Get the path to the cache file for a URL."""
        key = self._get_cache_key(url)
        return self.cache_dir / key

    def get(self, url: str) -> Optional[str]:
        """
        Get a cached response for a URL if it exists and is not expired.

        Args:
            url: The URL to get from cache

        Returns:
            The cached content or None if not in cache or expired
        """
        # First check memory cache
        if url in self.memory_cache:
            content, timestamp = self.memory_cache[url]
            if time.time() - timestamp <= self.max_age:
                return content
            # Remove expired item from memory cache
            del self.memory_cache[url]

        # Check disk cache
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            # Check if cache is expired
            if time.time() - cache_path.stat().st_mtime <= self.max_age:
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Add to memory cache
                    self.memory_cache[url] = (content, time.time())
                    return content
                except IOError as e:
                    logger.error(f"Failed to read cache file {cache_path}: {e}")
                    # Log stack trace for debugging
                    import traceback

                    logger.debug(f"Cache read error details: {traceback.format_exc()}")

            # Remove expired cache file
            try:
                cache_path.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove expired cache file {cache_path}: {e}")

        return None

    def set(self, url: str, content: str) -> None:
        """
        Cache a response for a URL.

        Args:
            url: The URL to cache
            content: The content to cache
        """
        # Calculate the size of the new content
        content_size = len(content.encode('utf-8'))

        # Check if the URL is already in the memory cache
        if url in self.memory_cache:
            old_content, _ = self.memory_cache[url]
            old_size = len(old_content.encode('utf-8'))
            # Adjust memory usage
            self.current_memory_usage = self.current_memory_usage - old_size + content_size
        else:
            # Add the new content size to the current usage
            self.current_memory_usage += content_size

            # Check if we need to make room in the memory cache
            self._check_memory_limits()

        # Update memory cache
        self.memory_cache[url] = (content, time.time())

        # Update disk cache
        cache_path = self._get_cache_path(url)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(content)
        except IOError as e:
            logger.warning(f"Failed to save response to cache: {e}")

    def _check_memory_limits(self) -> None:
        """Check if memory limits are exceeded and evict items if necessary."""
        # Check item count limit
        if len(self.memory_cache) >= self.max_memory_items:
            self._evict_lru_items(len(self.memory_cache) - self.max_memory_items + 1)

        # Check memory size limit
        max_bytes = self.max_memory_size_mb * 1024 * 1024
        if self.current_memory_usage > max_bytes:
            # Calculate how much we need to evict
            excess_bytes = self.current_memory_usage - max_bytes
            self._evict_bytes(excess_bytes)

    def _evict_lru_items(self, count: int) -> None:
        """Evict least recently used items from the memory cache."""
        if not self.memory_cache:
            return

        # Sort by timestamp (second element of the tuple)
        items = sorted(self.memory_cache.items(), key=lambda x: x[1][1])

        # Evict the oldest items
        for i in range(min(count, len(items))):
            url, (content, _) = items[i]
            content_size = len(content.encode('utf-8'))
            self.current_memory_usage -= content_size
            del self.memory_cache[url]

        logger.debug(f"Evicted {min(count, len(items))} items from memory cache")

    def _evict_bytes(self, bytes_to_evict: int) -> None:
        """Evict items from memory cache until the required bytes are freed."""
        if not self.memory_cache:
            return

        # Sort by timestamp (second element of the tuple)
        items = sorted(self.memory_cache.items(), key=lambda x: x[1][1])

        bytes_evicted = 0
        items_evicted = 0

        for url, (content, _) in items:
            content_size = len(content.encode('utf-8'))
            del self.memory_cache[url]
            bytes_evicted += content_size
            items_evicted += 1

            if bytes_evicted >= bytes_to_evict:
                break

        self.current_memory_usage -= bytes_evicted
        logger.debug(f"Evicted {items_evicted} items ({bytes_evicted} bytes) from memory cache")

    def clear(self, max_age: Optional[int] = None) -> int:
        """
        Clear expired cache entries.

        Args:
            max_age: Maximum age in seconds (defaults to instance max_age)

        Returns:
            Number of cache entries removed
        """
        if max_age is None:
            max_age = self.max_age

        # Clear memory cache
        current_time = time.time()
        expired_keys = [
            k
            for k, (_, timestamp) in self.memory_cache.items()
            if current_time - timestamp > max_age
        ]
        for k in expired_keys:
            # Subtract the content size from memory usage before removing
            content, _ = self.memory_cache[k]
            self.current_memory_usage -= len(content.encode('utf-8'))
            del self.memory_cache[k]

        # Clear disk cache
        count = 0
        for cache_file in self.cache_dir.glob("*"):
            if current_time - cache_file.stat().st_mtime > max_age:
                try:
                    cache_file.unlink()
                    count += 1
                except OSError as e:
                    logger.warning(f"Failed to clear cache file {cache_file}: {e}")

        return count + len(expired_keys)
