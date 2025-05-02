"""
Test error handling in the scraper module.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
import responses

from RAGnificent.core.cache import RequestCache
from RAGnificent.core.scraper import MarkdownScraper


class TestScraperErrorHandling(unittest.TestCase):
    """Test error handling in the scraper module."""

    def setUp(self):
        """Set up test environment with mocks and temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.chunk_dir = os.path.join(self.temp_dir, "chunks")
        self.cache_dir = os.path.join(self.temp_dir, "cache")

        # Create scraper instance for testing
        self.scraper = MarkdownScraper(
            requests_per_second=10.0,  # Fast for testing
            cache_enabled=True
        )

        # Replace the request cache with one using our temp directory
        self.scraper.request_cache = RequestCache(cache_dir=self.cache_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    @responses.activate
    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        url = "https://example.com/nonexistent"

        # Mock a connection error
        responses.add(
            responses.GET,
            url,
            body=requests.exceptions.ConnectionError("Connection refused")
        )

        # Attempt to scrape with max_retries=1 to speed up the test
        self.scraper.max_retries = 1

        with self.assertRaises(requests.exceptions.RequestException):
            self.scraper.scrape_website(url)

    @responses.activate
    def test_http_error_handling(self):
        """Test handling of HTTP errors."""
        url = "https://example.com/not-found"

        # Mock a 404 error
        responses.add(
            responses.GET,
            url,
            status=404
        )

        # Attempt to scrape with max_retries=1 to speed up the test
        self.scraper.max_retries = 1

        with self.assertRaises(requests.exceptions.HTTPError):
            self.scraper.scrape_website(url)

    @responses.activate
    def test_timeout_error_handling(self):
        """Test handling of timeout errors."""
        url = "https://example.com/slow-page"

        # Mock a timeout error
        responses.add(
            responses.GET,
            url,
            body=requests.exceptions.Timeout("Request timed out")
        )

        # Attempt to scrape with max_retries=1 to speed up the test
        self.scraper.max_retries = 1

        with self.assertRaises(requests.exceptions.RequestException):
            self.scraper.scrape_website(url)

    def test_file_io_error_handling(self):
        """Test handling of file I/O errors."""
        # Create an invalid directory path
        invalid_path = "/nonexistent/directory/file.md"

        # Mock a successful request
        with patch.object(self.scraper, 'convert_to_markdown', return_value="# Test\nContent"):
            with patch.object(self.scraper, 'scrape_website', return_value="<html><body><h1>Test</h1><p>Content</p></body></html>"):
                # Attempt to save to an invalid path
                with self.assertRaises(IOError):
                    self.scraper._process_single_url(
                        "https://example.com",
                        0,
                        1,
                        Path(invalid_path),
                        "markdown",
                        False,
                        None,
                        "jsonl"
                    )

    @patch('RAGnificent.core.scraper.MarkdownScraper.create_chunks')
    def test_chunking_error_handling(self, mock_create_chunks):
        """Test handling of errors during chunking."""
        # Mock an error during chunking
        mock_create_chunks.side_effect = Exception("Chunking error")

        # Create a valid output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Mock a successful request and conversion
        with patch.object(self.scraper, 'convert_to_markdown', return_value="# Test\nContent"):
            with patch.object(self.scraper, 'scrape_website', return_value="<html><body><h1>Test</h1><p>Content</p></body></html>"):
                # Attempt to process with chunking enabled
                try:
                    self.scraper._process_single_url(
                        "https://example.com",
                        0,
                        1,
                        Path(self.output_dir),
                        "markdown",
                        True,  # Save chunks
                        self.chunk_dir,
                        "jsonl"
                    )
                except Exception:
                    self.fail("Chunking error should be handled gracefully")

    @responses.activate
    def test_cache_error_handling(self):
        """Test handling of cache errors."""
        url = "https://example.com/cached-page"

        # Mock a successful response
        responses.add(
            responses.GET,
            url,
            body="<html><body><h1>Test</h1><p>Content</p></body></html>",
            status=200,
            content_type="text/html"
        )

        # Mock a cache error during storage
        with patch.object(self.scraper.request_cache, 'set', side_effect=IOError("Cache error")):
            # Should still return content even if caching fails
            content = self.scraper.scrape_website(url)
            self.assertIsNotNone(content, "Should return content even if caching fails")

    @responses.activate
    def test_parallel_processing_error_handling(self):
        """Test error handling with parallel processing."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a links file
        links_file = os.path.join(self.temp_dir, "links.txt")
        with open(links_file, "w") as f:
            f.write("https://example.com/good\nhttps://example.com/bad\n")

        # Mock responses for the URLs
        responses.add(
            responses.GET,
            "https://example.com/good",
            body="<html><body><h1>Good Page</h1><p>Content</p></body></html>",
            status=200,
            content_type="text/html"
        )

        # Error for the bad URL
        responses.add(
            responses.GET,
            "https://example.com/bad",
            status=500
        )

        # Process with parallel=True to test parallel error handling
        self.scraper.max_retries = 1  # Speed up test
        result = self.scraper.scrape_by_links_file(
            links_file=links_file,
            output_dir=self.output_dir,
            parallel=True,
            max_workers=2,
            worker_timeout=1  # Short timeout for testing
        )

        # Should have one successful URL
        self.assertEqual(len(result), 1, "Should handle errors in parallel processing")

    @responses.activate
    def test_worker_timeout_handling(self):
        """Test handling of worker timeouts in parallel processing."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Create a links file
        links_file = os.path.join(self.temp_dir, "links.txt")
        with open(links_file, "w") as f:
            f.write("https://example.com/fast\nhttps://example.com/slow\n")

        # Mock responses for the URLs
        responses.add(
            responses.GET,
            "https://example.com/fast",
            body="<html><body><h1>Fast Page</h1><p>Content</p></body></html>",
            status=200,
            content_type="text/html"
        )

        # Mock a very slow response that will trigger timeout
        def slow_response(request):
            import time
            time.sleep(2)  # This will exceed our worker_timeout
            return (200, {}, "<html><body><h1>Slow Page</h1></body></html>")

        responses.add_callback(
            responses.GET,
            "https://example.com/slow",
            callback=slow_response,
            content_type="text/html"
        )

        # Tests with a very short worker_timeout to simulate timeouts
        with patch('concurrent.futures.Future.result', side_effect=TimeoutError("Worker timeout")):
            result = self.scraper.scrape_by_links_file(
                links_file=links_file,
                output_dir=self.output_dir,
                parallel=True,
                max_workers=2,
                worker_timeout=1  # Very short timeout
            )

            # Should handle worker timeouts gracefully
            self.assertTrue(len(result) < 2, "Should handle worker timeouts")

    def test_memory_cache_limits(self):
        """Test that memory cache limits are enforced."""
        # Create a cache with very small limits
        small_cache = RequestCache(
            cache_dir=self.cache_dir,
            max_memory_items=2,  # Only 2 items allowed
            max_memory_size_mb=0.01  # Only 10KB allowed
        )

        # Add some items to fill the cache
        for i in range(5):
            # Create content larger than the memory limit
            content = f"Large content {i}" * 1000
            small_cache.set(f"http://example.com/page{i}", content)

        # Verify the memory usage is below the limit
        self.assertLessEqual(
            small_cache.current_memory_usage,
            0.01 * 1024 * 1024,  # 10KB
            "Memory cache should enforce size limits"
        )

        # Verify the item count is at most 2
        self.assertLessEqual(
            len(small_cache.memory_cache),
            2,
            "Memory cache should enforce item count limits"
        )


if __name__ == "__main__":
    unittest.main()
