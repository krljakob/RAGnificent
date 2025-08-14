"""
Cache module for HTTP requests to avoid repeated network calls.

Provides optimized caching with TTL, compression, and monitoring capabilities.
Enhanced with diskcache and joblib.Memory support for persistent caching.
"""

import gzip
import hashlib
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

try:
    import diskcache

    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

try:
    from joblib import Memory

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = logging.getLogger("request_cache")


try:
    from .stats import StatsMixin
except ImportError:
    from stats import StatsMixin


class RequestCache(StatsMixin):
    """HTTP request cache with TTL and compression."""

    def __init__(
        self,
        cache_dir: str = ".request_cache",
        max_age: int = 3600,
        max_memory_items: int = 100,
        max_memory_size_mb: int = 50,
        compression_threshold: int = 10240,  # 10KB
        enable_stats: bool = True,
        cache_backend: str = "auto",  # "auto", "diskcache", "joblib", "filesystem"
        diskcache_size_limit: int = 2**30,  # 1GB default
        joblib_compress: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age
        self.max_memory_items = max_memory_items
        self.max_memory_size_mb = max_memory_size_mb
        self.compression_threshold = compression_threshold
        super().__init__(enable_stats=enable_stats)

        self.cache_backend = self._select_cache_backend(cache_backend)
        self._init_cache_backend(diskcache_size_limit, joblib_compress)

        self.memory_cache: Dict[str, Tuple[str, float, Optional[int], bool]] = {}
        self.current_memory_usage = 0  # Approximate memory usage in bytes

        self.ttl_patterns: List[Tuple[Pattern, int]] = []

        logger.info(f"Initialized RequestCache with {self.cache_backend} backend")

        self.stats = {
            "hits": 0,
            "misses": 0,
            "memory_hits": 0,
            "disk_hits": 0,
            "sets": 0,
            "evictions": 0,
            "compression_savings": 0,
            "url_patterns": Counter(),
        }

        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        self._load_ttl_patterns()

    def _select_cache_backend(self, backend: str) -> str:
        if backend == "auto":
            if DISKCACHE_AVAILABLE:
                return "diskcache"
            return "joblib" if JOBLIB_AVAILABLE else "filesystem"
        if backend == "diskcache" and not DISKCACHE_AVAILABLE:
            logger.warning("diskcache not available, falling back to filesystem")
            return "filesystem"
        if backend == "joblib" and not JOBLIB_AVAILABLE:
            logger.warning("joblib not available, falling back to filesystem")
            return "filesystem"
        return backend

    def _init_cache_backend(
        self, diskcache_size_limit: int, joblib_compress: bool
    ) -> None:
        if self.cache_backend == "diskcache":
            self.disk_cache = diskcache.Cache(
                str(self.cache_dir / "diskcache"),
                size_limit=diskcache_size_limit,
                eviction_policy="least-recently-stored",
            )
            logger.info(
                f"Initialized diskcache with {diskcache_size_limit} bytes limit"
            )

        elif self.cache_backend == "joblib":
            self.memory_backend = Memory(
                location=str(self.cache_dir / "joblib"),
                compress=joblib_compress,
                verbose=0,
            )
            logger.info(f"Initialized joblib Memory with compression={joblib_compress}")

        else:  # filesystem
            self.disk_cache = None
            self.memory_backend = None
            logger.info("Using filesystem cache backend")

    def _load_ttl_patterns(self) -> None:
        ttl_file = self.metadata_dir / "ttl_patterns.json"
        if ttl_file.exists():
            try:
                with open(ttl_file, "r") as f:
                    patterns = json.load(f)

                for pattern_str, ttl in patterns.items():
                    self.add_ttl_pattern(pattern_str, ttl)

                logger.info(f"Loaded {len(patterns)} TTL patterns")
            except Exception as e:
                logger.error(f"Failed to load TTL patterns: {e}")

    def _save_ttl_patterns(self) -> None:
        """Save TTL patterns to metadata file."""
        ttl_file = self.metadata_dir / "ttl_patterns.json"
        try:
            patterns = {pattern.pattern: ttl for pattern, ttl in self.ttl_patterns}
            with open(ttl_file, "w") as f:
                json.dump(patterns, f)
        except Exception as e:
            logger.error(f"Failed to save TTL patterns: {e}")

    def add_ttl_pattern(self, pattern: str, ttl: int) -> None:
        """
        Add a URL pattern with a specific TTL.

        Args:
            pattern: Regex pattern to match URLs
            ttl: TTL in seconds for matching URLs
        """
        try:
            compiled = re.compile(pattern)
            self.ttl_patterns.append((compiled, ttl))
            self._save_ttl_patterns()
            logger.info(f"Added TTL pattern: {pattern} -> {ttl}s")
        except re.error as e:
            logger.error(f"Invalid regex pattern '{pattern}': {e}")

    def _get_ttl_for_url(self, url: str) -> Optional[int]:
        """Get the TTL for a URL based on patterns."""
        return next(
            (ttl for pattern, ttl in self.ttl_patterns if pattern.search(url)),
            None,
        )

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key from a URL."""
        return hashlib.blake2b(url.encode(), digest_size=16).hexdigest()

    def _get_cache_path(self, url: str) -> Path:
        """Get the path to the cache file for a URL."""
        key = self._get_cache_key(url)
        return self.cache_dir / key

    def _get_metadata_path(self, url: str) -> Path:
        """Get the path to the metadata file for a URL."""
        key = self._get_cache_key(url)
        return self.metadata_dir / f"{key}.meta"

    def _compress_content(self, content: str) -> Tuple[bytes, bool]:
        """
        Compress content if it exceeds the threshold.

        Returns:
            Tuple of (compressed_data, is_compressed)
        """
        encoded = content.encode("utf-8")
        if len(encoded) >= self.compression_threshold:
            compressed = gzip.compress(encoded)
            savings = len(encoded) - len(compressed)
            if self.enable_stats:
                self.stats["compression_savings"] += savings
            return compressed, True
        return encoded, False

    def _decompress_content(self, data: bytes, is_compressed: bool) -> str:
        if is_compressed:
            return gzip.decompress(data).decode("utf-8")
        return data.decode("utf-8")

    def get(self, url: str) -> Optional[str]:
        """Get cached response for URL if it exists and is not expired."""
        if self.enable_stats:
            for pattern, _ in self.ttl_patterns:
                if pattern.search(url):
                    self.stats["url_patterns"][pattern.pattern] += 1

        url_ttl = self._get_ttl_for_url(url) or self.max_age
        current_time = time.time()

        if url in self.memory_cache:
            content, timestamp, ttl, is_compressed = self.memory_cache[url]
            effective_ttl = ttl or url_ttl

            if current_time - timestamp <= effective_ttl:
                if self.enable_stats:
                    self.stats["hits"] += 1
                    self.stats["memory_hits"] += 1

                return (
                    self._decompress_content(content, True)
                    if is_compressed
                    else content
                )
            del self.memory_cache[url]
            logger.debug(f"Memory cache expired for {url}")

        cached_content = self._get_from_backend(url, url_ttl, current_time)
        if cached_content is not None:
            if self.enable_stats:
                self.stats["hits"] += 1
                self.stats["disk_hits"] += 1

            self._add_to_memory_cache(url, cached_content, current_time, url_ttl)
            return cached_content

        cache_path = self._get_cache_path(url)
        metadata_path = self._get_metadata_path(url)

        if cache_path.exists():
            cached_ttl = None
            is_compressed = False

            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        cached_ttl = metadata.get("ttl")
                        is_compressed = metadata.get("compressed", False)
                except Exception as e:
                    logger.warning(f"Failed to read cache metadata: {e}")

            effective_ttl = cached_ttl or url_ttl

            if current_time - cache_path.stat().st_mtime <= effective_ttl:
                try:
                    if is_compressed:
                        with open(cache_path, "rb") as f:
                            compressed_data = f.read()
                        content = self._decompress_content(compressed_data, True)
                    else:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            content = f.read()

                    if is_compressed:
                        self.memory_cache[url] = (
                            compressed_data,
                            current_time,
                            effective_ttl,
                            True,
                        )
                    else:
                        self.memory_cache[url] = (
                            content,
                            current_time,
                            effective_ttl,
                            False,
                        )

                    if self.enable_stats:
                        self.stats["hits"] += 1
                        self.stats["disk_hits"] += 1

                    return content
                except IOError as e:
                    logger.error(f"Failed to read cache file {cache_path}: {e}")
                    import traceback

                    logger.debug(f"Cache read error details: {traceback.format_exc()}")

            try:
                cache_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                logger.debug(f"Removed expired cache file for {url}")
            except OSError as e:
                logger.warning(f"Failed to remove expired cache file {cache_path}: {e}")

        if self.enable_stats:
            self.stats["misses"] += 1

        return None

    def _get_from_backend(
        self, url: str, url_ttl: int, current_time: float
    ) -> Optional[str]:
        cache_key = self._get_cache_key(url)

        if self.cache_backend == "diskcache":
            try:
                if cached_data := self.disk_cache.get(cache_key):
                    content, timestamp, ttl = cached_data
                    effective_ttl = ttl or url_ttl

                    if current_time - timestamp <= effective_ttl:
                        return content
                    del self.disk_cache[cache_key]
            except Exception as e:
                logger.warning(f"Error accessing diskcache: {e}")

        elif self.cache_backend == "joblib":
            try:
                # Joblib Memory doesn't have built-in TTL, so we store timestamp with data
                cached_func = self.memory_backend.cache(lambda key: None)
                if cached_data := cached_func.call_and_shelve(cache_key).get():
                    content, timestamp, ttl = cached_data
                    effective_ttl = ttl or url_ttl

                    if current_time - timestamp <= effective_ttl:
                        return content
                    # Clear expired item
                    cached_func.call_and_shelve(cache_key).clear()
            except Exception as e:
                logger.warning(f"Error accessing joblib cache: {e}")

        return None

    def _set_to_backend(
        self, url: str, content: str, ttl: Optional[int] = None
    ) -> None:
        """Store content in the configured backend cache."""
        cache_key = self._get_cache_key(url)
        timestamp = time.time()
        cached_data = (content, timestamp, ttl)

        if self.cache_backend == "diskcache":
            try:
                self.disk_cache[cache_key] = cached_data
            except Exception as e:
                logger.warning(f"Error storing to diskcache: {e}")

        elif self.cache_backend == "joblib":
            try:
                cached_func = self.memory_backend.cache(lambda key: cached_data)
                cached_func(cache_key)
            except Exception as e:
                logger.warning(f"Error storing to joblib cache: {e}")

    def _add_to_memory_cache(
        self, url: str, content: str, timestamp: float, ttl: Optional[int] = None
    ) -> None:
        """Add content to memory cache with size management."""
        # Check if compression is needed
        should_compress = len(content.encode("utf-8")) > self.compression_threshold
        if should_compress:
            compressed_content = self._compress_content(content)
            self.memory_cache[url] = (compressed_content, timestamp, ttl, True)
        else:
            self.memory_cache[url] = (content, timestamp, ttl, False)

        # Manage memory cache size
        self._check_memory_limits()

    def set(self, url: str, content: str, ttl: Optional[int] = None) -> None:
        """
        Cache a response for a URL.

        Args:
            url: The URL to cache
            content: The content to cache
            ttl: Time-to-live in seconds (overrides pattern-based TTL)
        """
        if self.enable_stats:
            self.stats["sets"] += 1

        if ttl is None:
            ttl = self._get_ttl_for_url(url)

        compressed_data, is_compressed = self._compress_content(content)

        content_size = len(compressed_data)

        if url in self.memory_cache:
            old_content, _, _, old_compressed = self.memory_cache[url]
            old_size = len(old_content)
            self.current_memory_usage = (
                self.current_memory_usage - old_size + content_size
            )
        else:
            self.current_memory_usage += content_size

            self._check_memory_limits()

        self._set_to_backend(url, content, ttl)

        current_time = time.time()
        if is_compressed:
            self.memory_cache[url] = (compressed_data, current_time, ttl, True)
        else:
            self.memory_cache[url] = (content, current_time, ttl, False)

        cache_path = self._get_cache_path(url)
        metadata_path = self._get_metadata_path(url)

        try:
            if is_compressed:
                with open(cache_path, "wb") as f:
                    f.write(compressed_data)
            else:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(content)

            with open(metadata_path, "w") as f:
                metadata = {
                    "url": url,
                    "timestamp": current_time,
                    "ttl": ttl,
                    "compressed": is_compressed,
                    "size": content_size,
                }
                json.dump(metadata, f)

        except IOError as e:
            logger.warning(f"Failed to save response to cache: {e}")

    def _check_memory_limits(self) -> None:
        if len(self.memory_cache) >= self.max_memory_items:
            self._evict_lru_items(len(self.memory_cache) - self.max_memory_items + 1)

        max_bytes = self.max_memory_size_mb * 1024 * 1024
        if self.current_memory_usage > max_bytes:
            excess_bytes = self.current_memory_usage - max_bytes
            self._evict_bytes(excess_bytes)

    def _evict_lru_items(self, count: int) -> None:
        if not self.memory_cache:
            return

        items = sorted(self.memory_cache.items(), key=lambda x: x[1][1])

        evicted = 0
        for i in range(min(count, len(items))):
            url, (content, _, _, _) = items[i]
            content_size = len(content)
            self.current_memory_usage -= content_size
            del self.memory_cache[url]
            evicted += 1

        if self.enable_stats:
            self.stats["evictions"] += evicted

        logger.debug(f"Evicted {evicted} items from memory cache")

    def _evict_bytes(self, bytes_to_evict: int) -> None:
        if not self.memory_cache:
            return

        items = sorted(self.memory_cache.items(), key=lambda x: x[1][1])

        bytes_evicted = 0
        items_evicted = 0

        for url, (content, _, _, _) in items:
            content_size = len(content)
            del self.memory_cache[url]
            bytes_evicted += content_size
            items_evicted += 1

            if bytes_evicted >= bytes_to_evict:
                break

        self.current_memory_usage -= bytes_evicted

        if self.enable_stats:
            self.stats["evictions"] += items_evicted

        logger.debug(
            f"Evicted {items_evicted} items ({bytes_evicted} bytes) from memory cache"
        )

    def clear(
        self, max_age: Optional[int] = None, pattern: Optional[str] = None
    ) -> int:
        """Clear expired cache entries or entries matching a pattern."""
        if max_age is None:
            max_age = self.max_age

        pattern_obj = None
        if pattern:
            try:
                pattern_obj = re.compile(pattern)
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {e}")
                return 0

        current_time = time.time()
        expired_keys = []

        for k, (content, timestamp, ttl, _) in self.memory_cache.items():
            if pattern_obj and not pattern_obj.search(k):
                continue

            effective_ttl = ttl or max_age
            if current_time - timestamp > effective_ttl:
                expired_keys.append(k)

        for k in expired_keys:
            # Subtract the content size from memory usage before removing
            content, _, _, _ = self.memory_cache[k]
            self.current_memory_usage -= len(content)
            del self.memory_cache[k]

        # Clear disk cache
        count = 0
        for cache_file in self.cache_dir.glob("*"):
            if cache_file.name == "metadata":
                continue

            # Check if this cache file matches the pattern
            if pattern_obj:
                metadata_path = self.metadata_dir / f"{cache_file.name}.meta"
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        url = metadata.get("url")
                        if url and not pattern_obj.search(url):
                            continue
                except Exception:
                    continue
            if pattern_obj or current_time - cache_file.stat().st_mtime > max_age:
                try:
                    cache_file.unlink()
                    metadata_path = self.metadata_dir / f"{cache_file.name}.meta"
                    if metadata_path.exists():
                        metadata_path.unlink()
                    count += 1
                except OSError as e:
                    logger.warning(f"Failed to clear cache file {cache_file}: {e}")

        return count + len(expired_keys)

    def invalidate(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.

        Args:
            pattern: Regex pattern to match URLs

        Returns:
            Number of cache entries invalidated
        """
        return self.clear(pattern=pattern)

    def preload(self, urls: List[str], content_getter: Callable[[str], str]) -> int:
        """
        Preload cache with content for specified URLs.

        Args:
            urls: List of URLs to preload
            content_getter: Function that takes a URL and returns content

        Returns:
            Number of URLs successfully preloaded
        """
        count = 0
        for url in urls:
            try:
                content = content_getter(url)
                self.set(url, content)
                count += 1
            except Exception as e:
                logger.error(f"Failed to preload {url}: {e}")

        return count

    def _get_stats_implementation(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """

        # Calculate hit rate
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        # Calculate memory usage
        memory_usage_mb = self.current_memory_usage / (1024 * 1024)
        max_memory_mb = self.max_memory_size_mb

        # Calculate disk usage
        disk_usage = sum(
            f.stat().st_size
            for f in self.cache_dir.glob("**/*")
            if f.is_file()
            and not str(f).startswith(str(self.metadata_dir))
            and f.suffix != ".meta"
        )
        disk_usage_mb = disk_usage / (1024 * 1024)

        memory_items = len(self.memory_cache)
        disk_items = sum(
            bool(_.is_file() and _.name != "metadata") for _ in self.cache_dir.glob("*")
        )

        compression_savings_mb = self.stats["compression_savings"] / (1024 * 1024)

        return {
            "hit_rate": hit_rate,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "memory_hits": self.stats["memory_hits"],
            "disk_hits": self.stats["disk_hits"],
            "sets": self.stats["sets"],
            "evictions": self.stats["evictions"],
            "memory_usage_mb": memory_usage_mb,
            "memory_usage_percent": (
                (memory_usage_mb / max_memory_mb) * 100 if max_memory_mb > 0 else 0
            ),
            "disk_usage_mb": disk_usage_mb,
            "memory_items": memory_items,
            "disk_items": disk_items,
            "compression_savings_mb": compression_savings_mb,
            "top_patterns": dict(self.stats["url_patterns"].most_common(5)),
            "cache_backend": self.cache_backend,
        }


def cached_function(cache: RequestCache, ttl: Optional[int] = None):
    """
    Decorator for caching function results using the enhanced RequestCache.

    Args:
        cache: RequestCache instance to use
        ttl: Time-to-live for cached results

    Usage:
        @cached_function(cache, ttl=3600)
        def expensive_function(arg1, arg2):
            # Some expensive computation
            return result
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hashlib.sha256(str(args + tuple(sorted(kwargs.items()))).encode()).hexdigest()}"

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                try:
                    return json.loads(cached_result)
                except json.JSONDecodeError:
                    # Fall back to string result
                    return cached_result

            # Compute result and cache it
            result = func(*args, **kwargs)

            # Serialize result for caching
            try:
                serialized = json.dumps(result, default=str)
            except (TypeError, ValueError):
                serialized = str(result)

            cache.set(cache_key, serialized, ttl)
            return result

        return wrapper

    return decorator


def create_cache(
    cache_dir: str = ".rag_cache",
    backend: str = "auto",
    max_age: int = 3600,
    size_limit_gb: float = 1.0,
) -> RequestCache:
    """
    Factory function to create a cache instance.

    Args:
        cache_dir: Directory for cache storage
        backend: Cache backend ("auto", "diskcache", "joblib", "filesystem")
        max_age: Default TTL in seconds
        size_limit_gb: Size limit in GB for disk cache

    Returns:
        Configured RequestCache instance
    """
    return RequestCache(
        cache_dir=cache_dir,
        cache_backend=backend,
        max_age=max_age,
        diskcache_size_limit=int(size_limit_gb * 1024**3),
        joblib_compress=True,
        max_memory_items=200,
        max_memory_size_mb=100,
        enable_stats=True,
    )
