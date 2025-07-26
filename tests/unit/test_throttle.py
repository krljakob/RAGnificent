"""
Comprehensive tests for the throttle module.
"""

import asyncio
import time
from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from RAGnificent.core.throttle import (
    AsyncRequestThrottler,
    DomainStats,
    RequestThrottler,
)

# Import performance testing utilities
try:
    from tests.utils.performance_testing import (
        assert_immediate_operation,
        assert_rate_limit_timing,
        PerformanceBudgets,
        TimingCategories,
    )
except ImportError:
    # Fallback for environments where test utilities aren't available
    def assert_immediate_operation(duration, description=""):
        assert duration < 0.05, f"{description}: expected < 0.05s, got {duration:.3f}s"
    
    def assert_rate_limit_timing(actual, expected, description=""):
        tolerance = expected * 0.12  # 12% tolerance to match TimingCategories.RATE_LIMIT_TOLERANCE
        assert expected - tolerance <= actual <= expected + tolerance, \
            f"{description}: expected {expected:.3f}s ±12%, got {actual:.3f}s"
    
    class PerformanceBudgets:
        THROTTLE_IMMEDIATE = 0.05
    
    class TimingCategories:
        RATE_LIMIT_TOLERANCE = 10.0


class TestDomainStats:
    """Test DomainStats dataclass."""

    def test_domain_stats_initialization(self):
        """Test that DomainStats initializes with correct defaults."""
        stats = DomainStats()

        assert stats.success_count == 0
        assert stats.error_count == 0
        assert stats.timeout_count == 0
        assert stats.total_response_time == 0.0
        assert len(stats.request_times) == 0
        assert len(stats.status_codes) == 0
        assert stats.last_error_time is None
        assert stats.consecutive_errors == 0
        assert stats.backoff_until is None

    def test_request_times_maxlen(self):
        """Test that request_times deque has correct maxlen."""
        stats = DomainStats()

        # Add more than 100 items
        for i in range(150):
            stats.request_times.append(i)

        # Should only keep last 100
        assert len(stats.request_times) == 100
        assert stats.request_times[0] == 50  # First 50 should be dropped


class TestRequestThrottler:
    """Test RequestThrottler class."""

    def test_initialization(self):
        """Test throttler initialization with various parameters."""
        throttler = RequestThrottler(
            requests_per_second=2.0,
            domain_specific_limits={"example.com": 1.0},
            max_workers=5,
            adaptive_throttling=False,
            max_retries=2,
            retry_delay=1.0,
        )

        assert throttler.default_rate_limit == 2.0
        assert throttler.min_interval == 0.5
        assert throttler.domain_limits == {"example.com": 1.0}
        assert throttler.max_workers == 5
        assert throttler.adaptive_throttling is False
        assert throttler.max_retries == 2
        assert throttler.retry_delay == 1.0

    def test_throttle_basic(self):
        """Test basic throttling functionality."""
        throttler = RequestThrottler(requests_per_second=10.0)  # 0.1s interval

        start_time = time.time()
        throttler.throttle()
        first_elapsed = time.time() - start_time

        # First request should go through immediately
        assert_immediate_operation(first_elapsed, "First throttle request")

        # Second request should be throttled
        start_time = time.time()
        throttler.throttle()
        second_elapsed = time.time() - start_time

        # Should wait approximately 0.1 seconds with improved precision
        assert_rate_limit_timing(second_elapsed, 0.1, "Second throttle request")

    def test_throttle_domain_specific(self):
        """Test domain-specific throttling."""
        throttler = RequestThrottler(
            requests_per_second=10.0,  # 0.1s default
            domain_specific_limits={"slow.com": 1.0},  # 1s for slow.com
        )

        # Request to slow.com should use domain-specific limit
        start_time = time.time()
        throttler.throttle("https://slow.com/page")
        first_elapsed = time.time() - start_time
        assert_immediate_operation(first_elapsed, "First domain-specific request")

        # Second request should wait ~1 second with improved tolerance
        start_time = time.time()
        throttler.throttle("https://slow.com/page")
        second_elapsed = time.time() - start_time
        assert_rate_limit_timing(second_elapsed, 1.0, "Domain-specific throttle delay")

    def test_backpressure(self):
        """Test backpressure mechanism."""
        throttler = RequestThrottler(max_workers=10)

        # Simulate high active requests
        throttler.active_requests = 9  # Above threshold (80% of 10 = 8)

        start_time = time.time()
        throttler.throttle()
        elapsed = time.time() - start_time

        # Should apply backpressure delay
        assert throttler.backpressure_delay > 0
        assert elapsed >= throttler.backpressure_delay * 0.95  # Reduced variance from 10% to 5% for better precision

    def test_release_updates_stats(self):
        """Test that release updates statistics correctly."""
        throttler = RequestThrottler()

        # Start a request
        throttler.throttle("https://example.com/page")
        assert throttler.active_requests == 1

        # Release with success
        throttler.release(
            url="https://example.com/page",
            status_code=200,
            response_time=0.5,
        )

        assert throttler.active_requests == 0

        stats = throttler.domain_stats["example.com"]
        assert stats.success_count == 1
        assert stats.error_count == 0
        assert stats.status_codes[200] == 1
        assert stats.total_response_time == 0.5

    def test_release_with_error(self):
        """Test release with error updates error statistics."""
        throttler = RequestThrottler()

        throttler.throttle("https://example.com/page")

        # Release with error
        error = Exception("Connection error")
        throttler.release(
            url="https://example.com/page",
            error=error,
            response_time=0.1,
        )

        stats = throttler.domain_stats["example.com"]
        assert stats.error_count == 1
        assert stats.consecutive_errors == 1
        assert stats.last_error_time is not None

    def test_backoff_on_consecutive_errors(self):
        """Test exponential backoff on consecutive errors."""
        throttler = RequestThrottler(retry_delay=1.0)

        # Simulate consecutive errors
        for i in range(3):
            throttler.throttle("https://example.com/page")
            throttler.release(
                url="https://example.com/page",
                error=Exception("Error"),
            )

        stats = throttler.domain_stats["example.com"]
        assert stats.consecutive_errors == 3
        assert stats.backoff_until is not None
        assert stats.backoff_until > time.time()

    def test_execute_success(self):
        """Test successful execution with throttling."""
        throttler = RequestThrottler()

        # Mock function that returns a response
        mock_func = Mock(return_value=Mock(status_code=200))

        result = throttler.execute(mock_func, "https://example.com/page", param="value")

        assert result is not None
        assert result.status_code == 200
        mock_func.assert_called_once_with("https://example.com/page", param="value")

        # Check stats were updated
        stats = throttler.domain_stats["example.com"]
        assert stats.success_count == 1

    def test_execute_with_retry(self):
        """Test execution with retry on failure."""
        throttler = RequestThrottler(max_retries=2, retry_delay=0.1)

        # Mock function that fails twice then succeeds
        mock_func = Mock(side_effect=[
            Exception("First failure"),
            Exception("Second failure"),
            Mock(status_code=200),
        ])

        result = throttler.execute(mock_func, "https://example.com/page")

        assert result is not None
        assert result.status_code == 200
        assert mock_func.call_count == 3

    def test_execute_rate_limit_retry(self):
        """Test execution with retry on 429 status."""
        throttler = RequestThrottler()

        # Mock function that returns 429 then 200
        mock_func = Mock(side_effect=[
            Mock(status_code=429, headers={}),
            Mock(status_code=200),
        ])

        result = throttler.execute(mock_func, "https://example.com/page")

        assert result is not None
        assert result.status_code == 200
        assert mock_func.call_count == 2

    def test_execute_parallel(self):
        """Test parallel execution."""
        throttler = RequestThrottler(max_workers=2)

        # Mock function that returns different results
        def mock_func(url):
            return Mock(status_code=200, url=url)

        urls = [
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/page3",
        ]

        results = throttler.execute_parallel(mock_func, urls)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.status_code == 200
            assert result.url == urls[i]

    def test_extract_domain(self):
        """Test domain extraction from URLs."""
        throttler = RequestThrottler()

        assert throttler._extract_domain("https://example.com/page") == "example.com"
        assert throttler._extract_domain("http://sub.example.com:8080/") == "sub.example.com:8080"
        assert throttler._extract_domain("invalid url") == "unknown"

    def test_get_domain_rate_limit(self):
        """Test getting domain-specific rate limits."""
        throttler = RequestThrottler(
            requests_per_second=2.0,
            domain_specific_limits={
                "slow.com": 1.0,
                "*.example.com": 0.5,
            },
        )

        # Exact match
        assert throttler._get_domain_rate_limit("slow.com") == 1.0

        # Wildcard match
        assert throttler._get_domain_rate_limit("sub.example.com") == 0.5
        assert throttler._get_domain_rate_limit("api.example.com") == 0.5

        # Default
        assert throttler._get_domain_rate_limit("other.com") == 2.0

    def test_adaptive_throttling(self):
        """Test adaptive rate limit adjustment."""
        throttler = RequestThrottler(adaptive_throttling=True)

        # Simulate high error rate to trigger significant rate limit adjustment
        domain = "slow.com"
        stats = throttler.domain_stats[domain]

        # Add some request times (requirement for adjustment)
        for _ in range(10):
            stats.request_times.append(1.0)
        
        # Set high error rate (> 10% threshold)
        stats.success_count = 8
        stats.error_count = 4  # 4/(8+4) = 33% error rate
        
        # Should adjust rate limit down due to high error rate
        original_limit = throttler._get_domain_rate_limit(domain)
        throttler._adjust_rate_limit(domain)

        # Verify rate limit was actually adjusted
        new_limit = throttler._get_domain_rate_limit(domain)
        assert domain in throttler.domain_limits, "Domain should be added to limits"
        assert throttler.domain_limits[domain] < original_limit, "Rate limit should be reduced due to high error rate"
        
        # Verify the adjustment is meaningful (not just a trivial change)
        reduction_factor = throttler.domain_limits[domain] / original_limit
        assert reduction_factor < 0.9, "Rate limit should be significantly reduced"
        assert reduction_factor > 0.1, "Rate limit should not be reduced to near-zero"

    def test_get_retry_after(self):
        """Test parsing Retry-After header."""
        throttler = RequestThrottler()

        # Test with numeric value
        response = Mock(headers={"Retry-After": "30"})
        assert throttler._get_retry_after(response) == 30.0

        # Test with missing header
        response = Mock(headers={})
        retry_after = throttler._get_retry_after(response)
        assert 5.0 <= retry_after <= 10.0  # Default range

    def test_get_stats(self):
        """Test statistics retrieval."""
        throttler = RequestThrottler()

        # Make some requests
        throttler.throttle("https://example.com/page")
        throttler.release("https://example.com/page", status_code=200, response_time=0.5)

        stats = throttler.get_stats()

        assert stats["total_requests"] == 1
        assert "domains" in stats
        assert "example.com" in stats["domains"]
        assert stats["domains"]["example.com"]["success_count"] == 1


class TestAsyncRequestThrottler:
    """Test AsyncRequestThrottler class."""

    def test_initialization(self):
        """Test async throttler initialization."""
        throttler = AsyncRequestThrottler(
            requests_per_second=2.0,
            domain_specific_limits={"example.com": 1.0},
            max_workers=5,
            adaptive_throttling=False,
        )

        assert throttler.base_rate_limit == 2.0
        assert throttler.domain_specific_limits == {"example.com": 1.0}
        assert throttler.max_workers == 5
        assert throttler.adaptive_throttling is False

    @pytest.mark.asyncio
    async def test_throttle_async(self):
        """Test async throttling."""
        throttler = AsyncRequestThrottler(requests_per_second=10.0)  # 0.1s interval

        # First request should go through quickly
        start_time = time.time()
        await throttler.throttle("https://example.com/page1")
        first_elapsed = time.time() - start_time
        assert first_elapsed < 0.05

        # Second request to same domain should be throttled
        start_time = time.time()
        await throttler.throttle("https://example.com/page2")
        second_elapsed = time.time() - start_time
        assert second_elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_domain_specific_async(self):
        """Test domain-specific async throttling."""
        throttler = AsyncRequestThrottler(
            requests_per_second=10.0,
            domain_specific_limits={"slow.com": 1.0},
        )

        # Different domains should not interfere
        await throttler.throttle("https://fast.com/page")
        await throttler.throttle("https://slow.com/page")

        # Second request to slow.com should wait
        start_time = time.time()
        await throttler.throttle("https://slow.com/page2")
        elapsed = time.time() - start_time
        assert elapsed >= 0.9

    @pytest.mark.asyncio
    async def test_record_response(self):
        """Test recording response statistics."""
        throttler = AsyncRequestThrottler()

        # Record successful response
        await throttler.record_response("https://example.com/page", 200, 0.5)

        stats = throttler._domain_stats["example.com"]
        assert stats.success_count == 1
        assert stats.total_response_time == 0.5
        assert stats.status_codes[200] == 1

    @pytest.mark.asyncio
    async def test_backoff_on_errors(self):
        """Test backoff mechanism on consecutive errors."""
        throttler = AsyncRequestThrottler()

        # Record multiple errors
        for _ in range(3):
            await throttler.record_response("https://example.com/page", 500, 0.1)

        stats = throttler._domain_stats["example.com"]
        assert stats.consecutive_errors == 3
        assert stats.backoff_until is not None

    @pytest.mark.asyncio
    async def test_adaptive_rate_limiting(self):
        """Test adaptive rate limiting based on errors."""
        throttler = AsyncRequestThrottler(
            requests_per_second=2.0,
            adaptive_throttling=True,
        )

        # Simulate high error rate
        domain = "error.com"
        for _ in range(10):
            await throttler.record_response(f"https://{domain}/page", 200, 0.1)
        for _ in range(5):
            await throttler.record_response(f"https://{domain}/page", 500, 0.1)

        # Rate limit should be reduced
        rate_limit = throttler._get_domain_rate_limit(domain)
        assert rate_limit < throttler.base_rate_limit

    def test_get_stats_async(self):
        """Test statistics retrieval for async throttler."""
        throttler = AsyncRequestThrottler()

        # Add some stats
        stats = throttler._domain_stats["example.com"]
        stats.success_count = 10
        stats.error_count = 2
        stats.total_response_time = 5.0
        stats.status_codes[200] = 10
        stats.status_codes[500] = 2

        result = throttler.get_stats()

        assert result["total_domains"] == 1
        assert "example.com" in result["domains"]

        domain_stats = result["domains"]["example.com"]
        assert domain_stats["success_count"] == 10
        assert domain_stats["error_count"] == 2
        assert domain_stats["success_rate"] == 10/12
        assert domain_stats["avg_response_time"] == 5.0/12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
