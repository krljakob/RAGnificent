"""
Additional unit tests to improve scraper module coverage.
"""

import json
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from RAGnificent.core.scraper import MarkdownScraper


@pytest.mark.unit
class TestScraperCoverage:
    """Additional tests to improve scraper coverage."""

    @pytest.fixture
    def scraper(self):
        """Create a scraper instance with mocked dependencies."""
        with patch("RAGnificent.core.scraper.RequestCache"):
            return MarkdownScraper(
                requests_per_second=10,
                timeout=5,
                max_retries=2,
                cache_enabled=False
            )

    def test_scraper_initialization_with_defaults(self):
        """Test scraper initialization with default parameters."""
        scraper = MarkdownScraper()
        assert scraper.timeout == 30
        assert scraper.max_retries == 3
        assert scraper.chunker.chunk_size == 1000
        assert scraper.chunker.chunk_overlap == 200

    def test_scraper_initialization_with_custom_params(self):
        """Test scraper initialization with custom parameters."""
        scraper = MarkdownScraper(
            requests_per_second=5.0,
            timeout=60,
            max_retries=5,
            chunk_size=500,
            chunk_overlap=100,
            cache_enabled=True,
            cache_max_age=7200,
            max_workers=20,
            adaptive_throttling=False
        )
        assert scraper.timeout == 60
        assert scraper.max_retries == 5
        assert scraper.chunker.chunk_size == 500
        assert scraper.chunker.chunk_overlap == 100
        assert scraper.max_workers == 20

    def test_scraper_with_rust_module_available(self):
        """Test scraper when Rust module is available."""
        with patch("RAGnificent.core.scraper.convert_html") as mock_convert:
            mock_convert.return_value = "# Markdown Content"
            
            scraper = MarkdownScraper()
            assert scraper.rust_available is True
            assert scraper.convert_html == mock_convert

    def test_scraper_without_rust_module(self):
        """Test scraper fallback when Rust module is not available."""
        with patch("RAGnificent.core.scraper.convert_html", side_effect=ImportError):
            scraper = MarkdownScraper()
            assert scraper.rust_available is False
            
            # Test fallback converter
            html = "<h1>Test</h1><p>Content</p>"
            result = scraper.convert_html(html, "http://example.com", "markdown")
            assert "Test" in result

    def test_scrape_with_different_output_formats(self, scraper):
        """Test scraping with markdown, JSON, and XML output formats."""
        mock_response = Mock()
        mock_response.text = "<h1>Title</h1><p>Content</p>"
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        
        with patch.object(scraper.session, "get", return_value=mock_response):
            # Test markdown format
            markdown = scraper.scrape_website("http://example.com", output_format="markdown")
            assert markdown is not None
            
            # Test JSON format
            json_output = scraper.scrape_website("http://example.com", output_format="json")
            parsed = json.loads(json_output)
            assert "title" in parsed or "headers" in parsed
            
            # Test XML format
            xml_output = scraper.scrape_website("http://example.com", output_format="xml")
            assert "<?xml" in xml_output or "<document>" in xml_output

    def test_scrape_with_retry_logic(self, scraper):
        """Test scraper retry logic on failure."""
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        
        mock_response_success = Mock()
        mock_response_success.text = "<h1>Success</h1>"
        mock_response_success.status_code = 200
        mock_response_success.headers = {"content-type": "text/html"}
        
        with patch.object(
            scraper.session, "get",
            side_effect=[mock_response_fail, mock_response_success]
        ):
            result = scraper.scrape_website("http://example.com")
            assert result is not None

    def test_scrape_with_max_retries_exceeded(self, scraper):
        """Test scraper when max retries are exceeded."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        
        scraper.max_retries = 2
        with patch.object(scraper.session, "get", return_value=mock_response):
            result = scraper.scrape_website("http://example.com")
            assert result is None

    def test_extract_metadata(self, scraper):
        """Test metadata extraction from HTML."""
        html = """
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test description">
            <meta name="keywords" content="test, keywords">
            <meta property="og:title" content="OG Title">
            <meta property="og:image" content="http://example.com/image.jpg">
        </head>
        <body><h1>Content</h1></body>
        </html>
        """
        
        metadata = scraper._extract_metadata(html, "http://example.com")
        assert metadata["title"] == "Test Page"
        assert metadata["description"] == "Test description"
        assert metadata["keywords"] == "test, keywords"
        assert metadata["og_title"] == "OG Title"
        assert metadata["og_image"] == "http://example.com/image.jpg"

    def test_clean_text(self, scraper):
        """Test text cleaning functionality."""
        text = "  Multiple   spaces   \n\n\nand\n\n\nlines   "
        cleaned = scraper._clean_text(text)
        assert cleaned == "Multiple spaces and lines"

    def test_save_to_file(self, scraper, tmp_path):
        """Test saving scraped content to file."""
        content = "# Test Content\n\nThis is a test."
        output_file = tmp_path / "output.md"
        
        scraper._save_to_file(content, str(output_file))
        assert output_file.exists()
        assert output_file.read_text() == content

    def test_save_chunks(self, scraper, tmp_path):
        """Test saving content chunks."""
        chunks = [
            {"content": "Chunk 1", "metadata": {"index": 0}},
            {"content": "Chunk 2", "metadata": {"index": 1}}
        ]
        
        scraper._save_chunks(chunks, str(tmp_path))
        
        # Check that chunk files were created
        chunk_files = list(tmp_path.glob("chunk_*.json"))
        assert len(chunk_files) == 2

    def test_scrape_multiple_urls_sequential(self, scraper):
        """Test scraping multiple URLs sequentially."""
        urls = ["http://example1.com", "http://example2.com"]
        
        mock_response = Mock()
        mock_response.text = "<h1>Test</h1>"
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        
        with patch.object(scraper.session, "get", return_value=mock_response):
            results = scraper.scrape_multiple_urls(urls, parallel=False)
            assert len(results) == 2
            assert all(result["success"] for result in results)

    def test_scrape_multiple_urls_parallel(self, scraper):
        """Test scraping multiple URLs in parallel."""
        urls = ["http://example1.com", "http://example2.com", "http://example3.com"]
        
        mock_response = Mock()
        mock_response.text = "<h1>Test</h1>"
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        
        with patch.object(scraper.session, "get", return_value=mock_response):
            results = scraper.scrape_multiple_urls(urls, parallel=True, max_workers=2)
            assert len(results) == 3
            assert all(result["success"] for result in results)

    def test_scrape_with_chunking(self, scraper):
        """Test scraping with content chunking."""
        mock_response = Mock()
        mock_response.text = "<h1>Title</h1>" + "<p>Long content. " * 100 + "</p>"
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        
        with patch.object(scraper.session, "get", return_value=mock_response):
            result, chunks = scraper.scrape_website(
                "http://example.com",
                return_chunks=True
            )
            assert result is not None
            assert chunks is not None
            assert len(chunks) > 0

    def test_domain_specific_throttling(self):
        """Test domain-specific rate limiting."""
        domain_limits = {
            "example.com": 0.5,  # 0.5 requests per second
            "fast.com": 10.0     # 10 requests per second
        }
        
        scraper = MarkdownScraper(
            domain_specific_limits=domain_limits,
            cache_enabled=False
        )
        
        assert scraper.throttler.domain_specific_limits == domain_limits

    def test_cache_integration(self):
        """Test cache integration with scraper."""
        with patch("RAGnificent.core.scraper.RequestCache") as mock_cache_class:
            mock_cache = Mock()
            mock_cache.get.return_value = "Cached content"
            mock_cache_class.return_value = mock_cache
            
            scraper = MarkdownScraper(cache_enabled=True)
            
            # Mock response for uncached request
            mock_response = Mock()
            mock_response.text = "<h1>Fresh content</h1>"
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            
            with patch.object(scraper.session, "get", return_value=mock_response):
                # First request should check cache
                result = scraper.scrape_website("http://example.com")
                assert result == "Cached content"

    def test_handle_non_html_content(self, scraper):
        """Test handling of non-HTML content types."""
        mock_response = Mock()
        mock_response.text = '{"key": "value"}'
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.elapsed = Mock(total_seconds=Mock(return_value=0.5))
        
        with patch.object(scraper.session, "get", return_value=mock_response):
            result = scraper.scrape_website("http://example.com/api.json")
            assert result is not None
            # Assert that the result is the raw JSON string for non-HTML content
            # The scraper returns raw text for non-HTML content types
            assert result == '{"key": "value"}'

    def test_request_timeout(self, scraper):
        """Test request timeout handling."""
        scraper.timeout = 1
        
        with patch.object(
            scraper.session, "get",
            side_effect=requests.Timeout("Request timed out")
        ):
            result = scraper.scrape_website("http://slow-site.com")
            assert result is None

    def test_connection_error_handling(self, scraper):
        """Test connection error handling."""
        with patch.object(
            scraper.session, "get",
            side_effect=requests.ConnectionError("Connection failed")
        ):
            result = scraper.scrape_website("http://unreachable.com")
            assert result is None

    def test_invalid_url_handling(self, scraper):
        """Test handling of invalid URLs."""
        result = scraper.scrape_website("not-a-valid-url")
        assert result is None
        
        result = scraper.scrape_website("")
        assert result is None