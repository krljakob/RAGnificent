"""
Performance testing utilities for consistent timing validation and regression detection.

This module provides standardized timing assertions and performance validation
helpers to improve test reliability and catch performance regressions effectively.
"""

import statistics
import time
from contextlib import contextmanager
from typing import Callable, List, Optional, Union


class PerformanceError(AssertionError):
    """Raised when performance assertions fail."""

    pass


def assert_timing_within(
    actual: float,
    expected: float,
    tolerance_pct: float = 10.0,
    description: str = "Timing assertion",
) -> None:
    """
    Assert that actual timing is within tolerance percentage of expected timing.

    Args:
        actual: The actual measured time in seconds
        expected: The expected time in seconds
        tolerance_pct: Allowed tolerance as percentage (e.g., 10.0 for ±10%)
        description: Description of what is being timed for error messages

    Raises:
        PerformanceError: If timing is outside tolerance bounds
    """
    tolerance = expected * (tolerance_pct / 100.0)
    min_expected = expected - tolerance
    max_expected = expected + tolerance

    if not (min_expected <= actual <= max_expected):
        deviation_pct = abs((actual - expected) / expected) * 100
        raise PerformanceError(
            f"{description} failed: expected {expected:.3f}s ±{tolerance_pct}% "
            f"({min_expected:.3f}s-{max_expected:.3f}s), got {actual:.3f}s "
            f"(deviation: {deviation_pct:.1f}%)"
        )


def assert_performance_budget(
    actual: float, max_duration: float, description: str = "Performance budget"
) -> None:
    """
    Assert that actual timing meets performance budget (maximum allowed duration).

    Args:
        actual: The actual measured time in seconds
        max_duration: Maximum allowed duration in seconds
        description: Description of the operation being measured

    Raises:
        PerformanceError: If timing exceeds budget
    """
    if actual > max_duration:
        excess_pct = ((actual - max_duration) / max_duration) * 100
        raise PerformanceError(
            f"{description} budget exceeded: max {max_duration:.3f}s, "
            f"got {actual:.3f}s (excess: {excess_pct:.1f}%)"
        )


def assert_timing_statistical(
    operation: Callable[[], None],
    expected: float,
    tolerance_pct: float = 10.0,
    samples: int = 5,
    description: str = "Statistical timing assertion",
) -> None:
    """
    Assert timing using statistical sampling to reduce test flakiness.

    Args:
        operation: Function to execute and time
        expected: Expected duration in seconds
        tolerance_pct: Allowed tolerance as percentage
        samples: Number of samples to collect
        description: Description for error messages

    Raises:
        PerformanceError: If median timing is outside tolerance bounds
    """
    timings = []
    for _ in range(samples):
        start_time = time.time()
        operation()
        elapsed = time.time() - start_time
        timings.append(elapsed)

    median_time = statistics.median(timings)
    assert_timing_within(
        median_time,
        expected,
        tolerance_pct,
        f"{description} (median of {samples} samples)",
    )


@contextmanager
def timing_context(description: str = "Operation"):
    """
    Context manager for timing operations with automatic reporting.

    Args:
        description: Description of the operation being timed

    Yields:
        dict: Dictionary that will contain 'duration' after context exits
    """
    result = {}
    start_time = time.time()
    try:
        yield result
    finally:
        result["duration"] = time.time() - start_time


class PerformanceBudgets:
    """Centralized performance budgets for consistent validation."""

    # rate limiting operation budgets
    RATE_LIMIT_CHECK = 0.01  # 10ms for rate limit check
    THROTTLE_IMMEDIATE = 0.05  # 50ms for immediate throttle pass
    THROTTLE_DELAY = 1.0  # 1s base delay + tolerance

    # cache operation budgets
    CACHE_GET = 0.01  # 10ms for cache retrieval
    CACHE_SET = 0.02  # 20ms for cache storage
    CACHE_CLEAR = 0.1  # 100ms for cache clear operation

    # network operation budgets (mocked)
    MOCK_REQUEST = 0.01  # 10ms for mocked network request
    MOCK_RESPONSE = 0.005  # 5ms for mocked response processing

    # security operation budgets
    INPUT_VALIDATION = 0.01  # 10ms for input validation
    DATA_SANITIZATION = 0.02  # 20ms for data sanitization

    @classmethod
    def assert_budget(cls, actual: float, budget: float, operation: str) -> None:
        """Assert that an operation meets its performance budget."""
        assert_performance_budget(actual, budget, f"{operation} performance budget")


class TimingCategories:
    """Standard timing categories with appropriate tolerance levels."""

    # critical timing operations - tight tolerance
    CRITICAL_TOLERANCE = 5.0  # ±5%

    # rate limiting operations - medium tolerance for system variations
    RATE_LIMIT_TOLERANCE = (
        12.0  # ±12% (slightly increased to handle system timing variations)
    )

    # integration test operations - relaxed tolerance
    INTEGRATION_TOLERANCE = 15.0  # ±15%

    # network operation simulations - very relaxed tolerance
    NETWORK_TOLERANCE = 20.0  # ±20%


def create_mock_delay_function(target_delay: float) -> Callable[[], None]:
    """
    Create a mock function that simulates a specific delay for testing.

    This is more deterministic than using time.sleep() in tests and allows
    for better control over timing behavior.

    Args:
        target_delay: The delay to simulate in seconds

    Returns:
        Function that simulates the delay when called
    """

    def mock_delay():
        # in real tests, this would be mocked to not actually delay
        # but to return a predictable timing result
        time.sleep(target_delay)

    return mock_delay


# convenience functions for common timing patterns
def assert_immediate_operation(
    duration: float, description: str = "Immediate operation"
) -> None:
    """Assert that an operation completes immediately (< 50ms)."""
    assert_performance_budget(
        duration, PerformanceBudgets.THROTTLE_IMMEDIATE, description
    )


def assert_rate_limit_timing(
    actual: float, expected: float, description: str = "Rate limit timing"
) -> None:
    """Assert rate limiting timing with appropriate tolerance."""
    assert_timing_within(
        actual, expected, TimingCategories.RATE_LIMIT_TOLERANCE, description
    )


def assert_critical_timing(
    actual: float, expected: float, description: str = "Critical timing"
) -> None:
    """Assert critical timing operations with tight tolerance."""
    assert_timing_within(
        actual, expected, TimingCategories.CRITICAL_TOLERANCE, description
    )
