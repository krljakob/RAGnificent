"""
Test edge cases in the sitemap_utils module.
"""

import tempfile
import unittest
import warnings
from unittest.mock import patch

import responses

from RAGnificent.utils.sitemap import get_sitemap_urls
from RAGnificent.utils.sitemap_utils import (
    SitemapParser,
    SitemapURL,
    discover_site_urls,
)


class TestSitemapEdgeCases(unittest.TestCase):
    """Test edge cases for the sitemap utilities."""

    def setUp(self):
        """Set up test environment with mocks and temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        # Suppress deprecation warnings from sitemap.py
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def tearDown(self):
        """Clean up test environment."""
        warnings.resetwarnings()

    @responses.activate
    def test_empty_sitemap(self):
        """Test handling of empty sitemap."""
        # Mock response with an empty sitemap
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            body="<?xml version='1.0' encoding='UTF-8'?><urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'></urlset>",
            status=200,
            content_type="application/xml",
        )

        parser = SitemapParser()
        urls = parser.parse_sitemap("https://example.com")

        self.assertEqual(len(urls), 0, "Empty sitemap should return empty list of URLs")

    @responses.activate
    def test_invalid_xml_sitemap(self):
        """Test handling of invalid XML in sitemap."""
        # Mock response with invalid XML
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            body="<?xml version='1.0' encoding='UTF-8'?><urlset>Invalid XML content</url>",
            status=200,
            content_type="application/xml",
        )

        parser = SitemapParser()
        urls = parser.parse_sitemap("https://example.com")

        self.assertEqual(len(urls), 0, "Invalid XML should be handled gracefully")

    @responses.activate
    def test_http_error_in_sitemap(self):
        """Test handling of HTTP errors when fetching sitemap."""
        # Mock 404 response
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            status=404,
        )

        # Also mock robots.txt to return 404
        responses.add(
            responses.GET,
            "https://example.com/robots.txt",
            status=404,
        )

        parser = SitemapParser()
        urls = parser.parse_sitemap("https://example.com")

        self.assertEqual(len(urls), 0, "HTTP errors should be handled gracefully")

    @responses.activate
    def test_robots_txt_with_multiple_sitemaps(self):
        """Test handling of robots.txt with multiple sitemap declarations."""
        # Mock robots.txt with multiple sitemaps
        robots_content = """
        User-agent: *
        Allow: /

        Sitemap: https://example.com/sitemap1.xml
        Sitemap: https://example.com/sitemap2.xml
        """
        responses.add(
            responses.GET,
            "https://example.com/robots.txt",
            body=robots_content,
            status=200,
            content_type="text/plain",
        )

        # Mock both sitemaps
        sitemap1_content = """<?xml version='1.0' encoding='UTF-8'?>
        <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
            <url>
                <loc>https://example.com/page1</loc>
            </url>
        </urlset>
        """
        responses.add(
            responses.GET,
            "https://example.com/sitemap1.xml",
            body=sitemap1_content,
            status=200,
            content_type="application/xml",
        )

        # Don't add sitemap2.xml response - it should handle the first valid sitemap only

        parser = SitemapParser()
        urls = parser.parse_sitemap("https://example.com")

        self.assertEqual(len(urls), 1, "Should handle multiple sitemaps in robots.txt")

    @responses.activate
    def test_deeply_nested_sitemap_indices(self):
        """Test handling of deeply nested sitemap indices."""
        # Mock sitemap index
        sitemap_index = """<?xml version='1.0' encoding='UTF-8'?>
        <sitemapindex xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
            <sitemap>
                <loc>https://example.com/sitemap_level1.xml</loc>
            </sitemap>
        </sitemapindex>
        """
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            body=sitemap_index,
            status=200,
            content_type="application/xml",
        )

        # Mock level 1 sitemap index
        sitemap_level1 = """<?xml version='1.0' encoding='UTF-8'?>
        <sitemapindex xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
            <sitemap>
                <loc>https://example.com/sitemap_level2.xml</loc>
            </sitemap>
        </sitemapindex>
        """
        responses.add(
            responses.GET,
            "https://example.com/sitemap_level1.xml",
            body=sitemap_level1,
            status=200,
            content_type="application/xml",
        )

        # Mock level 2 sitemap with actual URLs
        sitemap_level2 = """<?xml version='1.0' encoding='UTF-8'?>
        <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
            <url>
                <loc>https://example.com/deep-page</loc>
            </url>
        </urlset>
        """
        responses.add(
            responses.GET,
            "https://example.com/sitemap_level2.xml",
            body=sitemap_level2,
            status=200,
            content_type="application/xml",
        )

        parser = SitemapParser()
        urls = parser.parse_sitemap("https://example.com")

        self.assertEqual(len(urls), 1, "Should handle deeply nested sitemap indices")
        self.assertEqual(urls[0].loc, "https://example.com/deep-page")

    @responses.activate
    def test_mixed_url_formats(self):
        """Test handling of mixed URL formats in sitemaps."""
        # Mock sitemap with various URL formats
        sitemap_content = """<?xml version='1.0' encoding='UTF-8'?>
        <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
            <url>
                <loc>https://example.com/page1</loc>
            </url>
            <url>
                <loc>http://example.com/page2</loc>
            </url>
            <url>
                <loc>/relative-path</loc>
            </url>
            <url>
                <loc>//example.com/protocol-relative</loc>
            </url>
        </urlset>
        """
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            body=sitemap_content,
            status=200,
            content_type="application/xml",
        )

        parser = SitemapParser()
        urls = parser.parse_sitemap("https://example.com")

        # Should find the 2 proper URLs that match the domain
        matches = [url for url in urls if "example.com" in url.loc]
        self.assertEqual(len(matches), 2, "Should handle mixed URL formats")

    @responses.activate
    def test_filter_urls_with_patterns(self):
        """Test URL filtering with include and exclude patterns."""
        # Mock sitemap with various URLs
        sitemap_content = """<?xml version='1.0' encoding='UTF-8'?>
        <urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'>
            <url>
                <loc>https://example.com/blog/post1</loc>
            </url>
            <url>
                <loc>https://example.com/products/item1</loc>
            </url>
            <url>
                <loc>https://example.com/about</loc>
            </url>
            <url>
                <loc>https://example.com/blog/post2</loc>
            </url>
        </urlset>
        """
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            body=sitemap_content,
            status=200,
            content_type="application/xml",
        )

        parser = SitemapParser()
        parser.parse_sitemap("https://example.com")

        # Test include patterns
        filtered_urls = parser.filter_urls(include_patterns=["blog"])
        self.assertEqual(
            len(filtered_urls), 2, "Should include only URLs matching pattern"
        )

        # Test exclude patterns
        filtered_urls = parser.filter_urls(exclude_patterns=["blog"])
        self.assertEqual(len(filtered_urls), 2, "Should exclude URLs matching pattern")

        # Test both include and exclude
        filtered_urls = parser.filter_urls(
            include_patterns=["blog", "products"], exclude_patterns=["post2"]
        )
        self.assertEqual(
            len(filtered_urls), 2, "Should handle both include and exclude patterns"
        )

    @responses.activate
    def test_html_sitemap_handling(self):
        """Test handling of HTML sitemaps."""
        # Mock HTML sitemap
        html_sitemap = """
        <!DOCTYPE html>
        <html>
        <head><title>Sitemap</title></head>
        <body>
            <h1>Sitemap</h1>
            <ul>
                <li><a href="https://example.com/page1">Page 1</a></li>
                <li><a href="https://example.com/page2">Page 2</a></li>
                <li><a href="https://example.com/page3">Page 3</a></li>
            </ul>
        </body>
        </html>
        """
        responses.add(
            responses.GET,
            "https://example.com/sitemap.html",
            body=html_sitemap,
            status=200,
            content_type="text/html",
        )

        # Mock XML sitemap as 404 so it tries the HTML sitemap
        responses.add(
            responses.GET,
            "https://example.com/sitemap.xml",
            status=404,
        )

        # Mock robots.txt to point to HTML sitemap
        robots_content = """
        User-agent: *
        Allow: /

        Sitemap: https://example.com/sitemap.html
        """
        responses.add(
            responses.GET,
            "https://example.com/robots.txt",
            body=robots_content,
            status=200,
            content_type="text/plain",
        )

        # Test with the full implementation
        parser = SitemapParser()
        urls = parser.parse_sitemap("https://example.com")

        self.assertTrue(len(urls) > 0, "Should extract URLs from HTML sitemap")

    def test_compatibility_functions(self):
        """Test compatibility between old and new implementations."""
        # Patch both implementations to return predictable results
        with patch(
            "RAGnificent.utils.sitemap_utils.SitemapParser.parse_sitemap"
        ) as mock_parse, patch(
            "RAGnificent.utils.sitemap_utils.SitemapParser.filter_urls"
        ) as mock_filter:
            # Set up mock return values
            mock_urls = [
                SitemapURL(loc="https://example.com/page1"),
                SitemapURL(loc="https://example.com/page2"),
            ]
            mock_parse.return_value = mock_urls
            mock_filter.return_value = mock_urls

            # Test the legacy function
            old_result = get_sitemap_urls("https://example.com")

            # Test the new function
            new_result = discover_site_urls("https://example.com")

            # Both should return a list of strings
            self.assertIsInstance(
                old_result, list, "Legacy function should return a list"
            )
            self.assertIsInstance(
                new_result, list, "New function should return a list"
            )

            # The results should match
            self.assertEqual(
                sorted(old_result),
                sorted(new_result),
                "Legacy and new implementations should return the same results",
            )


if __name__ == "__main__":
    unittest.main()
