"""
Security utilities for RAGnificent.

Provides security-related functions for rate limiting, sensitive data
redaction, and other security features to protect the application
and its users.
"""

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import bleach

logger = logging.getLogger(__name__)



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

    sanitized = headers.copy()

    sensitive_headers = {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "proxy-authorization",
    }

    for header in sensitive_headers:
        if header.lower() in sanitized:
            sanitized[header] = "[REDACTED]"

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
