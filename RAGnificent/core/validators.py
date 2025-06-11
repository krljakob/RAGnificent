"""
Input validation utilities for RAGnificent.

Provides validation functions for user inputs, URLs, and other data
to ensure security and data integrity throughout the application.
"""

import logging
import re
import urllib.parse
from typing import Any, List, Optional, Callable

logger = logging.getLogger(__name__)


def _validate_with_logging(value: Any, validation_func: Callable, error_msg: str = "Validation failed") -> bool:
    """Helper function to validate with consistent error logging."""
    if not value:
        return False
    
    try:
        return validation_func(value)
    except Exception as e:
        logger.warning(f"{error_msg}: {e}")
        return False


def validate_url(url: str) -> bool:
    """
    Validate if a URL is properly formatted and uses supported protocols.

    Args:
        url: URL string to validate

    Returns:
        bool: True if URL is valid, False otherwise
    """
    if not url:
        logger.warning("Empty URL provided")
        return False

    try:
        parsed = urllib.parse.urlparse(url)

        if not all([parsed.scheme, parsed.netloc]):
            logger.warning(f"Invalid URL format: {url}")
            return False

        if parsed.scheme not in ["http", "https"]:
            logger.warning(f"Unsupported URL protocol: {parsed.scheme}")
            return False

        return True
    except Exception as e:
        logger.warning(f"URL validation error for {url}: {e}")
        return False


def sanitize_url(url: str) -> str:
    """
    Sanitize a URL by removing potentially harmful components.

    Args:
        url: URL string to sanitize

    Returns:
        str: Sanitized URL
    """
    if not url:
        return ""

    try:
        parsed = urllib.parse.urlparse(url)

        netloc = parsed.netloc
        if "@" in netloc:
            netloc = netloc.split("@")[1]

        return urllib.parse.urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                "",  # Remove fragment
            )
        )
    except Exception as e:
        logger.warning(f"URL sanitization error for {url}: {e}")
        return ""


def validate_file_path(
    path: str, allowed_extensions: Optional[List[str]] = None
) -> bool:
    """
    Validate if a file path is safe and has an allowed extension.

    Args:
        path: File path to validate
        allowed_extensions: List of allowed file extensions (without dot)

    Returns:
        bool: True if path is valid, False otherwise
    """
    if not path:
        return False

    if ".." in path or "~" in path:
        logger.warning(f"Potential path traversal attempt: {path}")
        return False

    if allowed_extensions:
        ext = path.split(".")[-1].lower() if "." in path else ""
        if ext not in allowed_extensions:
            logger.warning(f"Invalid file extension: {ext}")
            return False

    return True


def validate_regex_pattern(pattern: str) -> bool:
    """Validate if a regex pattern is valid and safe."""
    def _validate_regex(p):
        re.compile(p)
        if any(x in p for x in ["(.*)*", "(.+)+", "(a|a|a|a|a|a)"]):
            logger.warning(f"Potentially catastrophic regex pattern: {p}")
            return False
        return True
    
    return _validate_with_logging(pattern, _validate_regex, f"Invalid regex pattern: {pattern}")


def validate_html_content(content: str) -> bool:
    """
    Validate if HTML content is safe for processing.

    Args:
        content: HTML content to validate

    Returns:
        bool: True if content is valid, False otherwise
    """
    if not content:
        return False

    if "<html" not in content.lower() and "<body" not in content.lower():
        logger.warning("Content does not appear to be valid HTML")
        return False

    script_count = content.lower().count("<script")
    if script_count > 20:
        logger.warning(f"Suspicious number of script tags: {script_count}")
        return False

    return True


def validate_output_format(format_str: str) -> bool:
    """
    Validate if the output format is supported.

    Args:
        format_str: Output format string to validate

    Returns:
        bool: True if format is valid, False otherwise
    """
    valid_formats = ["markdown", "json", "xml"]

    if not format_str:
        return False

    format_lower = format_str.lower()
    if format_lower not in valid_formats:
        logger.warning(f"Unsupported output format: {format_str}")
        return False

    return True


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> bool:
    """
    Validate chunking parameters.

    Args:
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        bool: True if parameters are valid, False otherwise
    """
    if chunk_size <= 0:
        logger.warning(f"Invalid chunk size: {chunk_size}")
        return False

    if chunk_overlap < 0:
        logger.warning(f"Invalid chunk overlap: {chunk_overlap}")
        return False

    if chunk_overlap >= chunk_size:
        logger.warning(
            f"Chunk overlap ({chunk_overlap}) must be less than chunk size ({chunk_size})"
        )
        return False

    return True


def validate_rate_limit(rate_limit: float) -> bool:
    """
    Validate rate limit parameter.

    Args:
        rate_limit: Rate limit in requests per second

    Returns:
        bool: True if rate limit is valid, False otherwise
    """
    if rate_limit <= 0:
        logger.warning(f"Invalid rate limit: {rate_limit}")
        return False

    if rate_limit > 100:
        logger.warning(f"Unusually high rate limit: {rate_limit}")
        return False

    return True


class InputValidator:
    """Utility class for validating various inputs."""

    @staticmethod
    def validate_input(input_type: str, value: Any) -> bool:
        """
        Validate an input based on its type.

        Args:
            input_type: Type of input to validate
            value: Value to validate

        Returns:
            bool: True if input is valid, False otherwise
        """
        validators = {
            "url": validate_url,
            "file_path": validate_file_path,
            "regex": validate_regex_pattern,
            "html": validate_html_content,
            "output_format": validate_output_format,
            "chunk_params": lambda v: validate_chunk_params(
                v.get("chunk_size", 0), v.get("chunk_overlap", 0)
            ),
            "rate_limit": validate_rate_limit,
        }

        if input_type not in validators:
            logger.warning(f"Unknown input type: {input_type}")
            return False

        return validators[input_type](value)
