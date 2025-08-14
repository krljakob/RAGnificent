"""
Rate limiting module for HTTP requests.

Provides domain-specific rate limiting, adaptive throttling,
parallel request management, and backpressure mechanisms.
"""

import asyncio
import logging
import random
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("request_throttler")


@dataclass
class DomainStats:
    """Statistics for a specific domain."""

    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    total_response_time: float = 0.0
    request_times: deque = field(default_factory=lambda: deque(maxlen=100))
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    last_error_time: Optional[float] = None
    consecutive_errors: int = 0
    backoff_until: Optional[float] = None


try:
    from .stats import StatsMixin
except ImportError:
    from stats import StatsMixin


class RequestThrottler(StatsMixin):
    """Request throttler with domain-specific rate limiting."""

    def __init__(
        self,
        requests_per_second: float = 1.0,
        domain_specific_limits: Optional[Dict[str, float]] = None,
        max_workers: int = 10,
        adaptive_throttling: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        enable_stats: bool = True,
    ):
        self.default_rate_limit = max(0.1, requests_per_second)
        self.min_interval = 1.0 / self.default_rate_limit
        self.domain_limits = domain_specific_limits or {}
        self.max_workers = max_workers
        self.adaptive_throttling = adaptive_throttling
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        super().__init__(enable_stats=enable_stats)

        self.domain_last_request: Dict[str, float] = defaultdict(float)
        self.domain_stats: Dict[str, DomainStats] = defaultdict(DomainStats)

        self.last_request_time: float = 0.0
        self.request_count: int = 0
        self.active_requests: int = 0

        self.lock = threading.RLock()

        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.backpressure_threshold = max_workers * 0.8
        self.backpressure_delay = 0.0

        logger.info(
            f"Initialized RequestThrottler with default rate limit of {self.default_rate_limit} "
            f"requests/second and {max_workers} workers"
        )

    def throttle(self, url: Optional[str] = None) -> None:
        """Wait if necessary to maintain the rate limit."""
        with self.lock:
            current_time = time.time()

            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)

            if url:
                domain = self._extract_domain(url)
                domain_rate_limit = self._get_domain_rate_limit(domain)
                domain_interval = 1.0 / domain_rate_limit

                stats = self.domain_stats[domain]
                if stats.backoff_until and current_time < stats.backoff_until:
                    backoff_sleep = stats.backoff_until - current_time
                    logger.debug(
                        f"Domain {domain} in backoff mode, sleeping for {backoff_sleep:.2f}s"
                    )
                    time.sleep(backoff_sleep)

                time_since_domain_last = current_time - self.domain_last_request[domain]
                if time_since_domain_last < domain_interval:
                    sleep_time = domain_interval - time_since_domain_last
                    logger.debug(
                        f"Throttling request to {domain} for {sleep_time:.2f}s"
                    )
                    time.sleep(sleep_time)

                self.domain_last_request[domain] = time.time()

            if self.active_requests > self.backpressure_threshold:
                self.backpressure_delay = min(1.0, self.backpressure_delay + 0.1)
                logger.debug(
                    f"Applying backpressure: {self.active_requests}/{self.max_workers} "
                    f"active requests, delay={self.backpressure_delay:.2f}s"
                )
                time.sleep(self.backpressure_delay)
            else:
                self.backpressure_delay = max(0.0, self.backpressure_delay - 0.05)

            self.last_request_time = time.time()
            self.request_count += 1
            self.active_requests += 1

    def release(
        self,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time: Optional[float] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Release a request slot and update statistics."""
        with self.lock:
            self.active_requests = max(0, self.active_requests - 1)

            if url and self.enable_stats:
                domain = self._extract_domain(url)
                stats = self.domain_stats[domain]

                if error:
                    stats.error_count += 1
                    stats.consecutive_errors += 1
                    stats.last_error_time = time.time()

                    if stats.consecutive_errors > 1:
                        backoff_time = min(
                            60, self.retry_delay * (2 ** (stats.consecutive_errors - 1))
                        )
                        backoff_time *= 1 + random.random()
                        stats.backoff_until = time.time() + backoff_time
                        logger.warning(
                            f"Domain {domain} experiencing errors ({stats.consecutive_errors} consecutive). "
                            f"Backing off for {backoff_time:.2f}s"
                        )
                else:
                    stats.success_count += 1
                    stats.consecutive_errors = 0
                    stats.backoff_until = None

                    if status_code:
                        stats.status_codes[status_code] += 1

                    if response_time:
                        stats.total_response_time += response_time
                        stats.request_times.append(response_time)

                        if self.adaptive_throttling and len(stats.request_times) >= 5:
                            self._adjust_rate_limit(domain)

    def execute(self, func: Callable, url: str, *args, **kwargs) -> Any:
        """Execute a function with throttling and retry logic."""
        domain = self._extract_domain(url)
        retries = 0

        while retries <= self.max_retries:
            self.throttle(url)
            start_time = time.time()

            try:
                result = func(url, *args, **kwargs)
                response_time = time.time() - start_time

                status_code = getattr(result, "status_code", None)

                self.release(url, status_code=status_code, response_time=response_time)

                if status_code in (429, 503):
                    retry_after = self._get_retry_after(result)
                    logger.warning(
                        f"Rate limited by {domain} (status {status_code}), waiting {retry_after}s"
                    )
                    time.sleep(retry_after)
                    retries += 1
                    continue

                return result

            except Exception as e:
                response_time = time.time() - start_time
                self.release(url, error=e, response_time=response_time)

                if retries < self.max_retries:
                    retry_delay = self.retry_delay * (2**retries)
                    logger.warning(
                        f"Request to {url} failed with {type(e).__name__}: {str(e)}. "
                        f"Retrying in {retry_delay:.2f}s ({retries+1}/{self.max_retries})"
                    )
                    time.sleep(retry_delay)
                    retries += 1
                else:
                    logger.error(
                        f"Request to {url} failed after {self.max_retries} retries: {str(e)}"
                    )
                    raise
        return None

    def execute_parallel(
        self, func: Callable, urls: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Execute a function on multiple URLs in parallel with throttling.

        Args:
            func: Function to execute
            urls: List of URLs to process
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            List of results in the same order as the input URLs
        """
        futures = []

        for url in urls:
            future = self.executor.submit(self.execute, func, url, *args, **kwargs)
            futures.append(future)

        return [future.result() for future in futures]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            netloc = urlparse(url).netloc
            return netloc or "unknown"
        except Exception:
            return "unknown"

    def _get_domain_rate_limit(self, domain: str) -> float:
        """Get rate limit for a specific domain."""
        if domain in self.domain_limits:
            return self.domain_limits[domain]

        return next(
            (
                limit
                for pattern, limit in self.domain_limits.items()
                if pattern.startswith("*.") and domain.endswith(pattern[1:])
            ),
            self.default_rate_limit,
        )

    def _adjust_rate_limit(self, domain: str) -> None:
        """Adaptively adjust rate limit based on response times and errors."""
        if not self.adaptive_throttling:
            return

        stats = self.domain_stats[domain]

        if len(stats.request_times) < 5:
            return

        avg_response_time = sum(stats.request_times) / len(stats.request_times)

        total_requests = stats.success_count + stats.error_count
        error_rate = stats.error_count / total_requests if total_requests > 0 else 0

        current_limit = self._get_domain_rate_limit(domain)
        new_limit = current_limit

        if avg_response_time > 2.0:
            new_limit = max(0.1, current_limit * 0.8)
        elif avg_response_time < 0.5 and error_rate < 0.05:
            new_limit = min(10.0, current_limit * 1.2)

        if error_rate > 0.1:
            new_limit = max(0.1, current_limit * 0.5)

        if abs(new_limit - current_limit) / current_limit > 0.2:
            logger.info(
                f"Adjusting rate limit for {domain} from {current_limit:.2f} to {new_limit:.2f} "
                f"requests/s (avg_time={avg_response_time:.2f}s, error_rate={error_rate:.2f})"
            )
            self.domain_limits[domain] = new_limit

    def _get_retry_after(self, response: Any) -> float:
        """Extract Retry-After header or use default backoff."""
        retry_after = None

        if hasattr(response, "headers") and "Retry-After" in response.headers:
            retry_value = response.headers["Retry-After"]
            try:
                retry_after = float(retry_value)
            except ValueError:
                try:
                    retry_date = datetime.strptime(
                        retry_value, "%a, %d %b %Y %H:%M:%S %Z"
                    )
                    retry_after = (retry_date - datetime.now()).total_seconds()
                except ValueError:
                    pass

        if retry_after is None or retry_after <= 0:
            retry_after = 5.0 + random.random() * 5.0

        return retry_after

    def _get_stats_implementation(self) -> Dict[str, Any]:
        """
        Get throttling statistics.

        Returns:
            Dictionary of throttling statistics
        """

        with self.lock:
            stats = {
                "total_requests": self.request_count,
                "active_requests": self.active_requests,
                "backpressure_delay": self.backpressure_delay,
                "domains": {},
            }

            for domain, domain_stats in self.domain_stats.items():
                total = domain_stats.success_count + domain_stats.error_count
                success_rate = domain_stats.success_count / total if total > 0 else 0

                avg_response_time = 0
                if domain_stats.request_times:
                    avg_response_time = sum(domain_stats.request_times) / len(
                        domain_stats.request_times
                    )

                stats["domains"][domain] = {
                    "rate_limit": self._get_domain_rate_limit(domain),
                    "success_count": domain_stats.success_count,
                    "error_count": domain_stats.error_count,
                    "success_rate": success_rate,
                    "avg_response_time": avg_response_time,
                    "consecutive_errors": domain_stats.consecutive_errors,
                    "status_codes": dict(domain_stats.status_codes),
                }

            return stats


class AsyncRequestThrottler:
    """
    Async version of RequestThrottler using asyncio primitives.

    Features:
    - Domain-specific rate limiting with asyncio.Semaphore
    - Adaptive throttling based on server response
    - Async backoff and retry logic
    """

    def __init__(
        self,
        requests_per_second: float = 1.0,
        domain_specific_limits: Optional[Dict[str, float]] = None,
        max_workers: int = 10,
        adaptive_throttling: bool = True,
    ):
        """Initialize async throttler."""
        self.base_rate_limit = requests_per_second
        self.domain_specific_limits = domain_specific_limits or {}
        self.max_workers = max_workers
        self.adaptive_throttling = adaptive_throttling

        # Async synchronization primitives
        self._domain_locks = defaultdict(lambda: asyncio.Lock())
        self._domain_stats = defaultdict(DomainStats)
        self._last_request_times = defaultdict(float)

        # Global semaphore to limit concurrent requests
        self._global_semaphore = asyncio.Semaphore(max_workers)

    async def throttle(self, url: str) -> None:
        """
        Apply rate limiting for the given URL's domain.

        Args:
            url: The URL being requested
        """
        import asyncio

        domain = urlparse(url).netloc

        async with self._global_semaphore:
            async with self._domain_locks[domain]:
                await self._apply_domain_throttling(domain)

    async def _apply_domain_throttling(self, domain: str) -> None:
        """Apply throttling for a specific domain."""
        import asyncio

        domain_stats = self._domain_stats[domain]

        # Check if we're in a backoff period
        if domain_stats.backoff_until and time.time() < domain_stats.backoff_until:
            backoff_time = domain_stats.backoff_until - time.time()
            logger.info(
                f"Backing off for domain {domain} for {backoff_time:.2f} seconds"
            )
            await asyncio.sleep(backoff_time)
            domain_stats.backoff_until = None

        # Calculate rate limit for this domain
        rate_limit = self._get_domain_rate_limit(domain)
        min_interval = 1.0 / rate_limit

        # Calculate time since last request
        current_time = time.time()
        last_request_time = self._last_request_times[domain]
        time_since_last = current_time - last_request_time

        # Apply throttling if needed
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"Throttling domain {domain} for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)

        # Update last request time
        self._last_request_times[domain] = time.time()

    def _get_domain_rate_limit(self, domain: str) -> float:
        """Get rate limit for a specific domain."""
        if domain in self.domain_specific_limits:
            return self.domain_specific_limits[domain]

        if self.adaptive_throttling:
            domain_stats = self._domain_stats[domain]

            # Adjust rate based on error rate
            if domain_stats.success_count + domain_stats.error_count > 10:
                error_rate = domain_stats.error_count / (
                    domain_stats.success_count + domain_stats.error_count
                )

                if error_rate > 0.2:  # High error rate
                    return self.base_rate_limit * 0.5
                if error_rate > 0.1:  # Moderate error rate
                    return self.base_rate_limit * 0.75

        return self.base_rate_limit

    async def record_response(
        self, url: str, status_code: int, response_time: float
    ) -> None:
        """Record response statistics for adaptive throttling."""
        domain = urlparse(url).netloc
        domain_stats = self._domain_stats[domain]

        if 200 <= status_code < 300:
            domain_stats.success_count += 1
            domain_stats.consecutive_errors = 0
        else:
            domain_stats.error_count += 1
            domain_stats.consecutive_errors += 1
            domain_stats.last_error_time = time.time()

            # Apply exponential backoff for consecutive errors
            if domain_stats.consecutive_errors >= 3:
                backoff_time = min(60.0, 2 ** (domain_stats.consecutive_errors - 2))
                domain_stats.backoff_until = time.time() + backoff_time
                logger.warning(
                    f"Applying {backoff_time:.2f}s backoff for domain {domain} after {domain_stats.consecutive_errors} consecutive errors"
                )

        domain_stats.total_response_time += response_time
        domain_stats.request_times.append(time.time())
        domain_stats.status_codes[status_code] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get throttling statistics."""
        stats = {
            "total_domains": len(self._domain_stats),
            "base_rate_limit": self.base_rate_limit,
            "max_workers": self.max_workers,
            "domains": {},
        }

        for domain, domain_stats in self._domain_stats.items():
            total_requests = domain_stats.success_count + domain_stats.error_count
            success_rate = (
                domain_stats.success_count / total_requests if total_requests > 0 else 0
            )
            avg_response_time = (
                domain_stats.total_response_time / total_requests
                if total_requests > 0
                else 0
            )

            stats["domains"][domain] = {
                "rate_limit": self._get_domain_rate_limit(domain),
                "success_count": domain_stats.success_count,
                "error_count": domain_stats.error_count,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "consecutive_errors": domain_stats.consecutive_errors,
                "status_codes": dict(domain_stats.status_codes),
            }

        return stats
