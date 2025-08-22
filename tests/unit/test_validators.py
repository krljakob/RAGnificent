"""
Tests for the validators module.
"""

from unittest.mock import patch

import pytest

from RAGnificent.core.validators import (
    InputValidator,
    sanitize_url,
    validate_chunk_params,
    validate_file_path,
    validate_html_content,
    validate_output_format,
    validate_rate_limit,
    validate_regex_pattern,
    validate_url,
)


class TestValidateUrl:
    """Test validate_url function."""

    def test_valid_urls(self):
        """Test validation of valid URLs."""
        valid_urls = [
            "http://example.com",
            "https://www.example.com",
            "https://example.com/path/to/page",
            "http://example.com:8080",
            "https://subdomain.example.com",
            "https://example.com/path?query=value",
        ]

        for url in valid_urls:
            assert validate_url(url) is True, f"Failed for {url}"

    def test_invalid_urls(self):
        """Test validation of invalid URLs."""
        invalid_urls = [
            "",
            None,
            "not a url",
            "ftp://example.com",  # Unsupported protocol
            "javascript:alert()",  # Dangerous protocol
            "file:///etc/passwd",  # File protocol
            "//example.com",  # Missing protocol
            "http://",  # Missing domain
            "http:/example.com",  # Malformed
        ]

        for url in invalid_urls:
            assert validate_url(url) is False, f"Failed for {url}"

    def test_url_with_credentials(self):
        """Test URLs with credentials are accepted."""
        # URLs with credentials should be valid (sanitization removes them)
        assert validate_url("http://user:pass@example.com") is True

    def test_validation_behavior_for_edge_cases(self):
        """Test URL validation behavior for edge cases and error conditions."""
        # Test empty URL behavior
        result_empty = validate_url("")
        assert result_empty is False

        # Test unsupported protocol behavior
        result_ftp = validate_url("ftp://example.com")
        assert result_ftp is False

        # Test malformed URL behavior
        result_malformed = validate_url("http://")
        assert result_malformed is False

        # Test javascript protocol (security risk)
        result_js = validate_url("javascript:alert()")
        assert result_js is False

        # Test that validation correctly identifies these as invalid
        # rather than just checking if warnings were logged
        invalid_urls = [
            "",
            "ftp://example.com",
            "javascript:alert()",
            "file:///etc/passwd",
            "//example.com",
            "http://",
        ]

        for url in invalid_urls:
            assert validate_url(url) is False, f"URL should be invalid: {url}"


class TestSanitizeUrl:
    """Test sanitize_url function."""

    def test_basic_sanitization(self):
        """Test basic URL sanitization."""
        url = "https://example.com/path#fragment"
        sanitized = sanitize_url(url)

        # Fragment should be removed
        assert sanitized == "https://example.com/path"

    def test_credential_removal(self):
        """Test removal of credentials from URL."""
        url = "https://user:password@example.com/path"
        sanitized = sanitize_url(url)

        assert "user" not in sanitized
        assert "password" not in sanitized
        assert sanitized == "https://example.com/path"

    def test_empty_url(self):
        """Test sanitization of empty URL."""
        assert sanitize_url("") == ""
        assert sanitize_url(None) == ""

    def test_malformed_url(self):
        """Test sanitization of malformed URL."""
        # Malformed URLs may still be processed by urlparse
        sanitized_url = sanitize_url("not a url")
        # Should either return empty or the original string
        assert sanitized_url in {"", "not a url"}

    def test_preserve_query_params(self):
        """Test that query parameters are preserved."""
        url = "https://example.com/search?q=test&page=2"
        sanitized = sanitize_url(url)

        assert "q=test" in sanitized
        assert "page=2" in sanitized


class TestValidateFilePath:
    """Test validate_file_path function."""

    def test_valid_paths(self):
        """Test validation of valid file paths."""
        valid_paths = [
            "file.txt",
            "data/file.csv",
            "output/report.pdf",
            "my-file_name.doc",
        ]

        for path in valid_paths:
            assert validate_file_path(path) is True, f"Failed for {path}"

    def test_invalid_paths(self):
        """Test validation of invalid file paths."""
        invalid_paths = [
            "",
            None,
            "../etc/passwd",  # Path traversal
            "../../secret.txt",  # Path traversal
            "~/home/user/file",  # Home directory expansion
            "file/../../../etc/passwd",  # Path traversal
        ]

        for path in invalid_paths:
            assert validate_file_path(path) is False, f"Failed for {path}"

    def test_extension_validation(self):
        """Test file extension validation."""
        allowed_extensions = ["txt", "csv", "json"]

        # Valid extensions
        assert validate_file_path("data.txt", allowed_extensions) is True
        assert validate_file_path("report.csv", allowed_extensions) is True
        assert validate_file_path("config.json", allowed_extensions) is True

        # Invalid extensions
        assert validate_file_path("script.exe", allowed_extensions) is False
        assert validate_file_path("image.png", allowed_extensions) is False
        assert validate_file_path("no_extension", allowed_extensions) is False

    def test_case_insensitive_extensions(self):
        """Test that extension checking is case-insensitive."""
        allowed_extensions = ["txt", "csv"]

        assert validate_file_path("File.TXT", allowed_extensions) is True
        assert validate_file_path("Data.CSV", allowed_extensions) is True


class TestValidateRegexPattern:
    """Test validate_regex_pattern function."""

    def test_valid_patterns(self):
        """Test validation of valid regex patterns."""
        valid_patterns = [
            r"\d+",
            r"[a-zA-Z]+",
            r"^test.*$",
            r"(foo|bar)",
            r"\w+@\w+\.\w+",
        ]

        for pattern in valid_patterns:
            assert validate_regex_pattern(pattern) is True, f"Failed for {pattern}"

    def test_invalid_patterns(self):
        """Test validation of invalid regex patterns."""
        invalid_patterns = [
            "",
            None,
            "[",  # Unclosed bracket
            "((",  # Unmatched parentheses
            r"\k<undefined>",  # Invalid backreference
        ]

        for pattern in invalid_patterns:
            assert validate_regex_pattern(pattern) is False, f"Failed for {pattern}"

    def test_catastrophic_patterns(self):
        """Test detection of potentially catastrophic regex patterns."""
        dangerous_patterns = [
            r"(.*)*",  # Exponential backtracking
            r"(.+)+",  # Exponential backtracking
            r"(a|a|a|a|a|a)",  # Redundant alternation
        ]

        for pattern in dangerous_patterns:
            assert validate_regex_pattern(pattern) is False, f"Failed for {pattern}"

    def test_catastrophic_pattern_detection_behavior(self):
        """Test that catastrophic regex patterns are actually rejected."""
        # Test that the function correctly identifies and rejects dangerous patterns
        dangerous_patterns = [
            r"(.*)*",  # Exponential backtracking
            r"(.+)+",  # Exponential backtracking
            r"(a|a|a|a|a|a)",  # Redundant alternation
        ]

        for pattern in dangerous_patterns:
            is_valid = validate_regex_pattern(pattern)
            assert is_valid is False, f"Dangerous pattern should be rejected: {pattern}"

        # Test that normal patterns are still accepted
        safe_patterns = [
            r"\d+",
            r"[a-zA-Z]+",
            r"^test.*$",
            r"(foo|bar)",
        ]

        for pattern in safe_patterns:
            is_valid = validate_regex_pattern(pattern)
            assert is_valid is True, f"Safe pattern should be accepted: {pattern}"


class TestValidateHtmlContent:
    """Test validate_html_content function."""

    def test_valid_html(self):
        """Test validation of valid HTML content."""
        valid_html = [
            "<html><body>Hello</body></html>",
            "<HTML><BODY>Content</BODY></HTML>",
            "<!DOCTYPE html><html><head></head><body></body></html>",
            "<body>Simple content</body>",
        ]

        for html in valid_html:
            assert validate_html_content(html) is True, f"Failed for {html[:50]}"

    def test_invalid_html(self):
        """Test validation of invalid HTML content."""
        invalid_html = [
            "",
            None,
            "Just plain text",
            "{ 'json': 'data' }",
            "<div>Not really HTML</div>",
        ]

        for html in invalid_html:
            assert validate_html_content(html) is False, f"Failed for {html}"

    def test_suspicious_html(self):
        """Test detection of suspicious HTML with many script tags."""
        # Create HTML with 21 script tags (threshold is 20)
        suspicious_html = (
            "<html><body>" + "<script>alert()</script>" * 21 + "</body></html>"
        )

        assert validate_html_content(suspicious_html) is False

    def test_normal_script_count(self):
        """Test that normal number of script tags is allowed."""
        normal_html = (
            "<html><body>"
            + "<script>console.log('ok')</script>" * 10
            + "</body></html>"
        )

        assert validate_html_content(normal_html) is True


class TestValidateOutputFormat:
    """Test validate_output_format function."""

    def test_valid_formats(self):
        """Test validation of valid output formats."""
        valid_formats = ["markdown", "json", "xml", "Markdown", "JSON", "XML"]

        for fmt in valid_formats:
            assert validate_output_format(fmt) is True, f"Failed for {fmt}"

    def test_invalid_formats(self):
        """Test validation of invalid output formats."""
        invalid_formats = ["", None, "pdf", "html", "text", "yaml"]

        for fmt in invalid_formats:
            assert validate_output_format(fmt) is False, f"Failed for {fmt}"

    def test_format_validation_comprehensive_behavior(self):
        """Test output format validation behavior."""
        # Test valid formats (case insensitive)
        valid_formats = ["markdown", "json", "xml", "Markdown", "JSON", "XML"]
        for fmt in valid_formats:
            result = validate_output_format(fmt)
            assert result is True, f"Valid format should be accepted: {fmt}"

        # Test invalid formats
        invalid_formats = ["", None, "pdf", "html", "text", "yaml", "csv", "docx"]
        for fmt in invalid_formats:
            result = validate_output_format(fmt)
            assert result is False, f"Invalid format should be rejected: {fmt}"

        # Test edge cases
        edge_cases = [
            "MARKDOWN",  # All caps
            "Json",  # Mixed case
            "  xml  ",  # Whitespace (this will fail as expected)
            "json,xml",  # Multiple formats
        ]

        # Test actual behavior rather than logging
        assert validate_output_format("MARKDOWN") is True
        assert validate_output_format("Json") is True
        assert (
            validate_output_format("  xml  ") is False
        )  # Function doesn't strip whitespace
        assert validate_output_format("json,xml") is False


class TestValidateChunkParams:
    """Test validate_chunk_params function."""

    def test_valid_params(self):
        """Test validation of valid chunk parameters."""
        valid_params = [
            (1000, 100),  # Normal values
            (500, 0),  # No overlap
            (2000, 200),  # Larger values
            (100, 50),  # 50% overlap
        ]

        for chunk_size, chunk_overlap in valid_params:
            assert validate_chunk_params(chunk_size, chunk_overlap) is True

    def test_invalid_chunk_size(self):
        """Test validation of invalid chunk sizes."""
        invalid_sizes = [0, -1, -100]

        for size in invalid_sizes:
            assert validate_chunk_params(size, 0) is False

    def test_invalid_chunk_overlap(self):
        """Test validation of invalid chunk overlaps."""
        assert validate_chunk_params(1000, -1) is False
        assert validate_chunk_params(1000, -100) is False

    def test_overlap_exceeds_size(self):
        """Test that overlap cannot exceed chunk size."""
        assert validate_chunk_params(100, 100) is False  # Equal
        assert validate_chunk_params(100, 150) is False  # Greater

    def test_chunk_validation_edge_cases_and_boundaries(self):
        """Test chunk parameter validation with edge cases and boundary conditions."""
        # Test invalid chunk sizes
        invalid_sizes = [0, -1, -100, -1000]
        for size in invalid_sizes:
            result = validate_chunk_params(size, 0)
            assert result is False, f"Invalid chunk size should be rejected: {size}"

        # Test invalid overlaps
        invalid_overlaps = [-1, -10, -100]
        for overlap in invalid_overlaps:
            result = validate_chunk_params(1000, overlap)
            assert result is False, f"Invalid overlap should be rejected: {overlap}"

        # Test overlap greater than or equal to chunk size
        boundary_cases = [
            (100, 100),  # Equal - should be invalid
            (100, 150),  # Greater - should be invalid
            (50, 75),  # Greater - should be invalid
            (1, 1),  # Equal - should be invalid
        ]

        for size, overlap in boundary_cases:
            result = validate_chunk_params(size, overlap)
            assert (
                result is False
            ), f"Overlap >= size should be invalid: {size}, {overlap}"

        # Test valid boundary cases
        valid_cases = [
            (100, 99),  # Just under
            (1000, 500),  # 50% overlap
            (200, 50),  # 25% overlap
            (1000, 0),  # No overlap
            (1, 0),  # Minimal valid case
        ]

        for size, overlap in valid_cases:
            result = validate_chunk_params(size, overlap)
            assert result is True, f"Valid params should be accepted: {size}, {overlap}"


class TestValidateRateLimit:
    """Test validate_rate_limit function."""

    def test_valid_rates(self):
        """Test validation of valid rate limits."""
        valid_rates = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

        for rate in valid_rates:
            assert validate_rate_limit(rate) is True, f"Failed for {rate}"

    def test_invalid_rates(self):
        """Test validation of invalid rate limits."""
        invalid_rates = [0, -1, -10, 0.0]

        for rate in invalid_rates:
            assert validate_rate_limit(rate) is False, f"Failed for {rate}"

    def test_unusually_high_rates(self):
        """Test warning for unusually high rate limits."""
        # Rates over 100 should still be valid but log a warning
        assert validate_rate_limit(101) is False
        assert validate_rate_limit(1000) is False

    def test_rate_limit_boundary_validation_behavior(self):
        """Test rate limit validation with boundary testing."""
        # Test invalid rates (zero and negative)
        invalid_rates = [0, -1, -10, -0.5, 0.0]
        for rate in invalid_rates:
            result = validate_rate_limit(rate)
            assert result is False, f"Invalid rate should be rejected: {rate}"

        # Test unusually high rates (over threshold)
        high_rates = [101, 150, 1000, 999999]
        for rate in high_rates:
            result = validate_rate_limit(rate)
            assert result is False, f"Unusually high rate should be rejected: {rate}"

        # Test valid rates within acceptable bounds
        valid_rates = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        for rate in valid_rates:
            result = validate_rate_limit(rate)
            assert result is True, f"Valid rate should be accepted: {rate}"

        # Test boundary conditions
        boundary_cases = [
            (0.001, True),  # Very small but positive
            (100.0, True),  # At upper limit
            (100.1, False),  # Just over limit
            (0.0, False),  # Zero boundary
        ]

        for rate, expected in boundary_cases:
            result = validate_rate_limit(rate)
            assert (
                result is expected
            ), f"Boundary case failed: {rate} should be {expected}"


class TestInputValidator:
    """Test InputValidator class."""

    def test_validate_url_input(self):
        """Test URL input validation."""
        validator = InputValidator()

        assert validator.validate_input("url", "https://example.com") is True
        assert validator.validate_input("url", "not a url") is False

    def test_validate_file_path_input(self):
        """Test file path input validation."""
        validator = InputValidator()

        assert validator.validate_input("file_path", "data/file.txt") is True
        assert validator.validate_input("file_path", "../etc/passwd") is False

    def test_validate_regex_input(self):
        """Test regex input validation."""
        validator = InputValidator()

        assert validator.validate_input("regex", r"\d+") is True
        assert validator.validate_input("regex", "(.*)*") is False

    def test_validate_html_input(self):
        """Test HTML input validation."""
        validator = InputValidator()

        assert (
            validator.validate_input("html", "<html><body>Test</body></html>") is True
        )
        assert validator.validate_input("html", "plain text") is False

    def test_validate_output_format_input(self):
        """Test output format input validation."""
        validator = InputValidator()

        assert validator.validate_input("output_format", "json") is True
        assert validator.validate_input("output_format", "pdf") is False

    def test_validate_chunk_params_input(self):
        """Test chunk parameters input validation."""
        validator = InputValidator()

        assert (
            validator.validate_input(
                "chunk_params", {"chunk_size": 1000, "chunk_overlap": 100}
            )
            is True
        )
        assert (
            validator.validate_input(
                "chunk_params", {"chunk_size": 100, "chunk_overlap": 150}
            )
            is False
        )

    def test_validate_rate_limit_input(self):
        """Test rate limit input validation."""
        validator = InputValidator()

        assert validator.validate_input("rate_limit", 1.0) is True
        assert validator.validate_input("rate_limit", 0) is False

    def test_unknown_input_type(self):
        """Test handling of unknown input types."""
        validator = InputValidator()

        assert validator.validate_input("unknown_type", "value") is False

    def test_input_validator_comprehensive_behavior(self):
        """Test input validator behavior across all input types."""
        validator = InputValidator()

        # Test all supported input types with valid inputs
        valid_test_cases = [
            ("url", "https://example.com"),
            ("file_path", "data/file.txt"),
            ("regex", r"\d+"),
            ("html", "<html><body>Test</body></html>"),
            ("output_format", "json"),
            ("chunk_params", {"chunk_size": 1000, "chunk_overlap": 100}),
            ("rate_limit", 1.0),
        ]

        for input_type, value in valid_test_cases:
            result = validator.validate_input(input_type, value)
            assert result is True, f"Valid {input_type} should be accepted: {value}"

        # Test all supported input types with invalid inputs
        invalid_test_cases = [
            ("url", "not a url"),
            ("file_path", "../etc/passwd"),
            ("regex", "(.*)*"),  # Catastrophic backtracking
            ("html", "plain text"),
            ("output_format", "pdf"),
            ("chunk_params", {"chunk_size": 100, "chunk_overlap": 150}),
            ("rate_limit", 0),
        ]

        for input_type, value in invalid_test_cases:
            result = validator.validate_input(input_type, value)
            assert result is False, f"Invalid {input_type} should be rejected: {value}"

        # Test unknown input type behavior
        result = validator.validate_input("unknown_type", "value")
        assert result is False, "Unknown input type should be rejected"

        # Test edge cases for chunk_params
        # Note: Missing chunk_overlap defaults to 0, which is valid
        chunk_edge_cases = [
            ({"chunk_size": 1000, "chunk_overlap": 100}, True),
            ({"chunk_size": 0, "chunk_overlap": 0}, False),  # Invalid chunk_size
            ({"chunk_size": 100}, True),  # Missing chunk_overlap defaults to 0 (valid)
            ({}, False),  # Empty dict - chunk_size defaults to 0 (invalid)
        ]

        for params, expected in chunk_edge_cases:
            result = validator.validate_input("chunk_params", params)
            assert result is expected, f"Chunk params edge case failed: {params}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
