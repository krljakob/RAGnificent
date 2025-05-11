"""
Cache module for HTTP requests to avoid repeated network calls.

Provides optimized caching with TTL, compression, and monitoring capabilities.
"""

import gzip
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

logger = logging.getLogger("request_cache")


class RequestCache:
    """
    Advanced cache for HTTP requests with TTL, compression, and monitoring.
    
    Features:
    - Memory and disk caching with configurable limits
    - Per-URL TTL configuration
    - Content compression for large responses
    - Cache statistics and monitoring
    - Pattern-based cache invalidation
    - Cache preloading for frequently accessed URLs
    """

    def __init__(
        self,
        cache_dir: str = ".request_cache",
        max_age: int = 3600,
        max_memory_items: int = 100,
        max_memory_size_mb: int = 50,
        compression_threshold: int = 10240,  # 10KB
        enable_stats: bool = True,
    ):
        """
        Initialize the request cache.

        Args:
            cache_dir: Directory to store cached responses
            max_age: Maximum age of cached responses in seconds (default: 1 hour)
            max_memory_items: Maximum number of items to keep in memory cache
            max_memory_size_mb: Maximum size of memory cache in MB
            compression_threshold: Minimum size in bytes for compressing content
            enable_stats: Whether to collect cache statistics
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age
        self.max_memory_items = max_memory_items
        self.max_memory_size_mb = max_memory_size_mb
        self.compression_threshold = compression_threshold
        self.enable_stats = enable_stats
        
        self.memory_cache: Dict[str, Tuple[str, float, Optional[int], bool]] = {}
        self.current_memory_usage = 0  # Approximate memory usage in bytes
        
        self.ttl_patterns: List[Tuple[Pattern, int]] = []
        
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

    def _load_ttl_patterns(self) -> None:
        """Load TTL patterns from metadata file."""
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
        for pattern, ttl in self.ttl_patterns:
            if pattern.search(url):
                return ttl
        return None
    
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
        """Decompress content if it was compressed."""
        if is_compressed:
            return gzip.decompress(data).decode("utf-8")
        return data.decode("utf-8")

    def get(self, url: str) -> Optional[str]:
        """
        Get a cached response for a URL if it exists and is not expired.

        Args:
            url: The URL to get from cache

        Returns:
            The cached content or None if not in cache or expired
        """
        if self.enable_stats:
            for pattern, _ in self.ttl_patterns:
                if pattern.search(url):
                    self.stats["url_patterns"][pattern.pattern] += 1
        
        url_ttl = self._get_ttl_for_url(url) or self.max_age
        current_time = time.time()
        
        # First check memory cache
        if url in self.memory_cache:
            content, timestamp, ttl, is_compressed = self.memory_cache[url]
            effective_ttl = ttl or url_ttl
            
            if current_time - timestamp <= effective_ttl:
                if self.enable_stats:
                    self.stats["hits"] += 1
                    self.stats["memory_hits"] += 1
                
                if is_compressed:
                    return self._decompress_content(content, True)
                return content
                
            # Remove expired item from memory cache
            del self.memory_cache[url]
            logger.debug(f"Memory cache expired for {url}")

        # Check disk cache
        cache_path = self._get_cache_path(url)
        metadata_path = self._get_metadata_path(url)
        
        if cache_path.exists():
            # Check metadata for TTL if available
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
            
            # Check if cache is expired
            if current_time - cache_path.stat().st_mtime <= effective_ttl:
                try:
                    if is_compressed:
                        with open(cache_path, "rb") as f:
                            compressed_data = f.read()
                        content = self._decompress_content(compressed_data, True)
                    else:
                        with open(cache_path, "r", encoding="utf-8") as f:
                            content = f.read()
                    
                    # Add to memory cache
                    if is_compressed:
                        self.memory_cache[url] = (compressed_data, current_time, effective_ttl, True)
                    else:
                        self.memory_cache[url] = (content, current_time, effective_ttl, False)
                    
                    if self.enable_stats:
                        self.stats["hits"] += 1
                        self.stats["disk_hits"] += 1
                    
                    return content
                except IOError as e:
                    logger.error(f"Failed to read cache file {cache_path}: {e}")
                    import traceback
                    logger.debug(f"Cache read error details: {traceback.format_exc()}")

            # Remove expired cache files
            try:
                cache_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
                logger.debug(f"Removed expired cache file for {url}")
            except OSError as e:
                logger.warning(f"Failed to remove expired cache file {cache_path}: {e}")

        if self.enable_stats:
            self.stats["misses"] += 1
            
        return None

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
        
        # Calculate the size of the new content (compressed or not)
        content_size = len(compressed_data)
        
        # Check if the URL is already in the memory cache
        if url in self.memory_cache:
            old_content, _, _, old_compressed = self.memory_cache[url]
            old_size = len(old_content)
            # Adjust memory usage
            self.current_memory_usage = (
                self.current_memory_usage - old_size + content_size
            )
        else:
            # Add the new content size to the current usage
            self.current_memory_usage += content_size

            # Check if we need to make room in the memory cache
            self._check_memory_limits()

        # Update memory cache with compressed data if applicable
        current_time = time.time()
        if is_compressed:
            self.memory_cache[url] = (compressed_data, current_time, ttl, True)
        else:
            self.memory_cache[url] = (content, current_time, ttl, False)

        # Update disk cache
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
        """Evict items from memory cache until the required bytes are freed."""
        if not self.memory_cache:
            return

        # Sort by timestamp (second element of the tuple)
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

    def clear(self, max_age: Optional[int] = None, pattern: Optional[str] = None) -> int:
        """
        Clear expired cache entries or entries matching a pattern.

        Args:
            max_age: Maximum age in seconds (defaults to instance max_age)
            pattern: Optional regex pattern to match URLs for selective clearing

        Returns:
            Number of cache entries removed
        """
        if max_age is None:
            max_age = self.max_age
            
        pattern_obj = None
        if pattern:
            try:
                pattern_obj = re.compile(pattern)
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {e}")
                return 0

        # Clear memory cache
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
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            url = metadata.get("url")
                            if url and not pattern_obj.search(url):
                                continue
                    except Exception:
                        continue
                else:
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
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        if not self.enable_stats:
            return {"stats_disabled": True}
            
        # Calculate hit rate
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        # Calculate memory usage
        memory_usage_mb = self.current_memory_usage / (1024 * 1024)
        max_memory_mb = self.max_memory_size_mb
        
        # Calculate disk usage
        disk_usage = sum(f.stat().st_size for f in self.cache_dir.glob("**/*") if f.is_file())
        disk_usage_mb = disk_usage / (1024 * 1024)
        
        memory_items = len(self.memory_cache)
        disk_items = sum(1 for _ in self.cache_dir.glob("*") if _.is_file() and _.name != "metadata")
        
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
            "memory_usage_percent": (memory_usage_mb / max_memory_mb) * 100 if max_memory_mb > 0 else 0,
            "disk_usage_mb": disk_usage_mb,
            "memory_items": memory_items,
            "disk_items": disk_items,
            "compression_savings_mb": compression_savings_mb,
            "top_patterns": dict(self.stats["url_patterns"].most_common(5)),
        }
