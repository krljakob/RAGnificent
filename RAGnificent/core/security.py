"""
Security utilities for RAGnificent.

Provides security-related functions for rate limiting, sensitive data
redaction, and other security features to protect the application
and its users.
"""

import logging
import re
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import bleach

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter for controlling request frequency."""

    def __init__(self, max_calls: int, time_frame: float):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed in the time frame
            time_frame: Time frame in seconds
        """
        self.max_calls = max_calls
        self.time_frame = time_frame
        self.calls = []

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator for rate-limiting function calls.

        Args:
            func: Function to rate limit

        Returns:
            Callable: Rate-limited function
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            self.calls = [
                call_time
                for call_time in self.calls
                if current_time - call_time <= self.time_frame
            ]

            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_frame - (current_time - self.calls[0])
                if sleep_time > 0:
                    logger.info(
                        f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds"
                    )
                    time.sleep(sleep_time)

            self.calls.append(time.time())
            return func(*args, **kwargs)

        return wrapper


class ThrottledSession:
    """Session with built-in throttling for HTTP requests."""

    def __init__(self, requests_per_second: float = 1.0):
        """
        Initialize throttled session.

        Args:
            requests_per_second: Maximum number of requests per second
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0

    def request(self, method: str, url: str, **kwargs) -> Any:
        """
        Make a throttled HTTP request.

        Args:
            method: HTTP method
            url: URL to request
            **kwargs: Additional arguments for requests

        Returns:
            Response from the request
        """
        import requests

        current_time = time.time()
        elapsed = current_time - self.last_request_time
        wait_time = max(0, self.min_interval - elapsed)

        if wait_time > 0:
            logger.debug(f"Throttling request to {url}. Waiting {wait_time:.2f}s")
            time.sleep(wait_time)

        self.last_request_time = time.time()
        return requests.request(method, url, **kwargs)


def redact_sensitive_data(
    text: str, patterns: Optional[List[Tuple[str, str]]] = None
) -> str:
    """
    Redact sensitive data from text.

    Args:
        text: Text to redact
        patterns: List of (pattern, replacement) tuples

    Returns:
        str: Redacted text
    """
    if not text:
        return ""

    default_patterns = [
        (
            r'(api[_-]?key|token)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{20,})["\']?',
            r"\1: [REDACTED]",
        ),
        (r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL REDACTED]"),
        (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP REDACTED]"),
        (r"(https?://)([^:@/]+):([^@/]+)@", r"\1[USER REDACTED]:[PASS REDACTED]@"),
        (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[CARD REDACTED]"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
        (r"AKIA[0-9A-Z]{16}", "[AWS KEY REDACTED]"),
        (
            r'(password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^"\'\s]{8,})["\']?',
            r"\1: [REDACTED]",
        ),
    ]

    patterns = patterns or default_patterns

    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)

    return result


def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Sanitize HTTP headers by removing sensitive information.

    Args:
        headers: HTTP headers dictionary

    Returns:
        Dict[str, str]: Sanitized headers
    """
    if not headers:
        return {}

    # Build a mapping of lowercase -> original key to preserve original casing
    original_keys: Dict[str, str] = {k.lower(): k for k in headers}

    sensitive_headers = {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "proxy-authorization",
    }

    sanitized: Dict[str, str] = headers.copy()

    for sensitive in sensitive_headers:
        if sensitive in original_keys:
            original_key = original_keys[sensitive]
            sanitized[original_key] = "[REDACTED]"

    return sanitized


def secure_file_path(base_dir: str, user_path: str) -> str:
    """
    Create a secure file path that prevents directory traversal.

    Args:
        base_dir: Base directory
        user_path: User-provided path

    Returns:
        str: Secure absolute path
    """
    import os

    base = os.path.abspath(base_dir)

    # Use pathlib for more secure path handling
    from pathlib import Path

    # Normalize and resolve the user path while preventing traversal
    try:
        user_path_obj = Path(user_path)
        # Remove any path traversal components
        clean_parts = [
            part for part in user_path_obj.parts if part not in ("..", ".", "~")
        ]
        clean_path = str(Path(*clean_parts)) if clean_parts else ""
    except (ValueError, OSError):
        logger.warning(f"Invalid path format: {user_path}")
        return base

    full_path = os.path.abspath(os.path.join(base, clean_path))

    if not full_path.startswith(base):
        logger.warning(f"Path traversal attempt detected: {user_path}")
        return base

    return full_path


def validate_content_security(content: str) -> bool:
    """
    Validate content for security issues.

    Args:
        content: Content to validate

    Returns:
        bool: True if content is safe, False otherwise
    """
    if not content:
        return True

    dangerous_patterns = [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"onerror=",
        r"onload=",
        r"eval\(",
        r"document\.cookie",
        r"<iframe",
        r"<object",
        r"<embed",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            logger.warning(f"Potentially unsafe content detected: {pattern}")
            return False

    return True


def sanitize_content(content: str) -> str:
    """
    Sanitize content by removing potentially dangerous elements.

    Args:
        content: Content to sanitize

    Returns:
        str: Sanitized content
    """
    if not content:
        return ""

    # Use bleach to sanitize content
    return bleach.clean(
        content,
        tags=[],  # Remove all HTML tags
        attributes={},  # Remove all attributes
        protocols=[],  # Remove all protocols
        strip=True,  # Strip disallowed tags instead of escaping
    )


class SecurityAuditLogger:
    """Logger for security-related events."""

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize security audit logger.

        Args:
            log_file: Path to log file
        """
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)

        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """
        Log a security event.

        Args:
            event_type: Type of security event
            details: Event details
        """
        safe_details = {k: redact_sensitive_data(str(v)) for k, v in details.items()}

        self.logger.info(f"Security event: {event_type} - {safe_details}")
