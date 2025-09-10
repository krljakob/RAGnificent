"""
Async version of the web scraper using httpx for improved performance.
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from RAGnificent.core.cache import RequestCache
from RAGnificent.core.logging import get_logger
from RAGnificent.core.throttle import AsyncRequestThrottler
from RAGnificent.utils.chunk_utils import ContentChunker, create_semantic_chunks
from RAGnificent.utils.sitemap_utils import SitemapParser

logger = get_logger(__name__)


class AsyncMarkdownScraper:
    """Async scraper for websites with conversion to markdown, JSON, or XML."""

    # precompiled regex patterns for better performance
    _whitespace_pattern = re.compile(r"\s+")
    _url_path_pattern = re.compile(r'[\\/*?:"<>|]')

    def __init__(
        self,
        requests_per_second: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cache_enabled: bool = True,
        cache_max_age: int = 3600,
        domain_specific_limits: Optional[Dict[str, float]] = None,
        max_workers: int = 10,
        adaptive_throttling: bool = True,
    ) -> None:
        """
        Initialize async scraper with httpx client.

        Args:
            requests_per_second: Maximum number of requests per second
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed requests
            chunk_size: Maximum size of content chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            cache_enabled: Whether to enable request caching
            cache_max_age: Maximum age of cached responses in seconds
            domain_specific_limits: Dict mapping domains to their rate limits
            max_workers: Maximum number of parallel workers
            adaptive_throttling: Whether to adjust rate limits based on responses
        """
        # configure httpx client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(
                max_connections=max_workers * 2, max_keepalive_connections=max_workers
            ),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            },
            follow_redirects=True,
        )

        self.throttler = AsyncRequestThrottler(
            requests_per_second=requests_per_second,
            domain_specific_limits=domain_specific_limits,
            max_workers=max_workers,
            adaptive_throttling=adaptive_throttling,
        )

        self.timeout = timeout
        self.max_retries = max_retries
        self.chunker = ContentChunker(chunk_size, chunk_overlap)
        self.max_workers = max_workers

        self.cache_enabled = cache_enabled
        self.request_cache = (
            RequestCache(max_age=cache_max_age) if cache_enabled else None
        )

        # try to use the Rust implementation if available
        try:
            from RAGnificent.ragnificent_rs import OutputFormat, convert_html

            self.rust_available = True
            self.OutputFormat = OutputFormat
            self.convert_html = convert_html
        except ImportError:
            self.rust_available = False

            # define fallback OutputFormat enum-like class
            class FallbackOutputFormat:
                MARKDOWN = "markdown"
                JSON = "json"
                XML = "xml"

            self.OutputFormat = FallbackOutputFormat

            # define fallback convert_html function
            def fallback_convert_html(html_content, url, output_format):
                from markdownify import markdownify

                return markdownify(html_content, heading_style="ATX")

            self.convert_html = fallback_convert_html

    async def scrape_website(self, url: str, skip_cache: bool = False) -> str:
        """
        Scrape a website asynchronously with retry logic, rate limiting, and caching.

        Args:
            url: The URL to scrape
            skip_cache: Whether to skip the cache and force a new request

        Returns:
            The HTML content as a string

        Raises:
            httpx.HTTPError: If the request fails after retries
            ValueError: If the URL is invalid
        """
        from RAGnificent.core.security import redact_sensitive_data
        from RAGnificent.core.validators import sanitize_url, validate_url

        if not validate_url(url):
            error_msg = f"Invalid URL format: {redact_sensitive_data(url)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        sanitized_url = sanitize_url(url)
        if sanitized_url != url:
            logger.warning(
                f"URL sanitized for security: {redact_sensitive_data(url)} -> {redact_sensitive_data(sanitized_url)}"
            )
            url = sanitized_url

        cached_content = self._check_cache(url, skip_cache)
        if cached_content is not None:
            return cached_content

        logger.info(f"Attempting to scrape the website: {redact_sensitive_data(url)}")

        try:
            # attempt to fetch content with retries
            html_content = await self._fetch_with_retries(url)

            # cache the response if enabled
            self._cache_response(url, html_content)

            logger.info(
                f"Successfully scraped content from: {redact_sensitive_data(url)}"
            )
            return html_content

        except Exception as e:
            logger.error(f"Failed to scrape {redact_sensitive_data(url)}: {str(e)}")
            raise

    async def _fetch_with_retries(self, url: str) -> str:
        """Fetch URL content with retry logic using async httpx."""
        for attempt in range(self.max_retries):
            try:
                await self.throttler.throttle(url)

                start_time = time.time()
                response = await self.client.get(url)
                response.raise_for_status()

                elapsed = time.time() - start_time
                logger.info(
                    f"Successfully retrieved the website content (status code: {response.status_code})."
                )
                logger.info(f"Network latency: {elapsed:.2f} seconds")

                return response.text

            except httpx.HTTPStatusError as e:
                await self._handle_request_error(
                    url,
                    attempt,
                    e,
                    f"HTTP error on attempt {attempt + 1}/{self.max_retries}: {e}",
                    f"Failed to retrieve {url} after {self.max_retries} attempts.",
                )
            except httpx.ConnectError as e:
                await self._handle_request_error(
                    url,
                    attempt,
                    e,
                    f"Connection error on attempt {attempt + 1}/{self.max_retries}: {e}",
                    f"Connection error persisted for {url} after {self.max_retries} attempts.",
                )
            except httpx.TimeoutException as e:
                await self._handle_request_error(
                    url,
                    attempt,
                    e,
                    f"Timeout on attempt {attempt + 1}/{self.max_retries}: {e}",
                    f"Request to {url} timed out after {self.max_retries} attempts.",
                )
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                raise

        raise httpx.HTTPError(
            f"Failed to retrieve {url} after {self.max_retries} attempts"
        )

    async def _handle_request_error(
        self, url: str, attempt: int, error, warning_msg: str, error_msg: str
    ) -> None:
        """Handle request errors with appropriate logging and retries."""
        logger.warning(warning_msg)

        # if this is the last attempt, log error and raise
        if attempt == self.max_retries - 1:
            logger.error(error_msg)
            raise error

        # otherwise apply exponential backoff
        await asyncio.sleep(2**attempt)

    async def scrape_multiple_urls(
        self,
        urls: List[str],
        output_dir: str = "output",
        output_format: str = "markdown",
        save_chunks: bool = False,
        chunk_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Scrape multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape
            output_dir: Directory to save output files
            output_format: Output format (markdown, json, xml)
            save_chunks: Whether to save content chunks
            chunk_dir: Directory to save chunks (if different from output_dir)

        Returns:
            List of successfully scraped URLs
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if save_chunks and chunk_dir:
            Path(chunk_dir).mkdir(parents=True, exist_ok=True)

        tasks = []
        for url in urls:
            task = self._scrape_and_save(
                url, output_dir, output_format, save_chunks, chunk_dir
            )
            tasks.append(task)

        semaphore = asyncio.Semaphore(self.max_workers)

        async def bounded_scrape(task):
            async with semaphore:
                return await task

        bounded_tasks = [bounded_scrape(task) for task in tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)

        # filter successful results
        successful_urls = []
        for url, result in zip(urls, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Failed to scrape {url}: {result}")
            else:
                successful_urls.append(url)

        return successful_urls

    async def _scrape_and_save(
        self,
        url: str,
        output_dir: str,
        output_format: str = "markdown",
        save_chunks: bool = False,
        chunk_dir: Optional[str] = None,
    ) -> None:
        """Scrape a single URL and save the output."""
        try:
            # scrape the website
            html_content = await self.scrape_website(url)

            if self.rust_available:
                format_map = {
                    "markdown": self.OutputFormat.MARKDOWN,
                    "json": self.OutputFormat.JSON,
                    "xml": self.OutputFormat.XML,
                }
                output = self.convert_html(
                    html_content,
                    url,
                    format_map.get(output_format, self.OutputFormat.MARKDOWN),
                )
            else:
                output = self._convert_content(html_content, url, output_format)

            parsed_url = urlparse(url)
            path_parts = parsed_url.path.strip("/").split("/")
            if path_parts and path_parts[-1]:
                base_name = self._url_path_pattern.sub("_", path_parts[-1])
            else:
                base_name = self._url_path_pattern.sub("_", parsed_url.netloc)

            allowed_extensions = {"markdown": "md", "json": "json", "xml": "xml"}
            extension = allowed_extensions.get(output_format, "md")
            output_file = Path(output_dir) / f"{base_name}.{extension}"

            # secure path validation using pathlib
            try:
                normalized_output_file = output_file.resolve()
                allowed_dir = Path(output_dir).resolve()

                # check if the resolved path is within the allowed directory using relative_to
                try:
                    normalized_output_file.relative_to(allowed_dir)
                except ValueError as e:
                    raise ValueError(
                        "Attempted to write outside the allowed directory."
                    ) from e
            except (OSError, ValueError) as e:
                raise ValueError(f"Invalid output path: {e}") from e

            with open(normalized_output_file, "w", encoding="utf-8") as f:
                f.write(output)

            logger.info(f"Saved {output_format} to: {output_file}")

            if save_chunks and output_format == "markdown":
                chunks = self.chunker.create_chunks_from_markdown(
                    output, source_url=url
                )
                chunks_output_dir = chunk_dir or output_dir

                for i, chunk in enumerate(chunks):
                    chunk_file = (
                        Path(chunks_output_dir) / f"{base_name}_chunk_{i:03d}.md"
                    )

                    # secure path validation using pathlib
                    try:
                        normalized_chunk_file = chunk_file.resolve()
                        allowed_chunks_dir = Path(chunks_output_dir).resolve()

                        # check if the resolved path is within the allowed directory using relative_to
                        try:
                            normalized_chunk_file.relative_to(allowed_chunks_dir)
                        except ValueError as exc:
                            raise ValueError(
                                "Attempted to write chunk outside the allowed directory."
                            ) from exc
                    except (OSError, ValueError) as e:
                        raise ValueError(f"Invalid chunk path: {e}") from e

                    with open(normalized_chunk_file, "w", encoding="utf-8") as f:
                        f.write(chunk["content"])

                logger.info(f"Saved {len(chunks)} chunks to: {chunks_output_dir}")

        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            raise

    async def scrape_by_sitemap(
        self,
        url: str,
        output_dir: str = "output",
        output_format: str = "markdown",
        min_priority: float = 0.0,
        url_filter: Optional[str] = None,
        save_chunks: bool = False,
        chunk_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Scrape all URLs from a sitemap concurrently.

        Args:
            url: Base URL or sitemap URL
            output_dir: Directory to save output files
            output_format: Output format (markdown, json, xml)
            min_priority: Minimum priority for URLs to scrape
            url_filter: Regex pattern to filter URLs
            save_chunks: Whether to save content chunks
            chunk_dir: Directory to save chunks

        Returns:
            List of successfully scraped URLs
        """
        sitemap_parser = SitemapParser()
        sitemap_urls = sitemap_parser.parse_sitemap(
            url, min_priority=min_priority, url_filter=url_filter
        )

        if not sitemap_urls:
            logger.warning("No URLs found in sitemap")
            return []

        logger.info(f"Found {len(sitemap_urls)} URLs in sitemap")

        # scrape all URLs concurrently
        return await self.scrape_multiple_urls(
            [sitemap_url.url for sitemap_url in sitemap_urls],
            output_dir=output_dir,
            output_format=output_format,
            save_chunks=save_chunks,
            chunk_dir=chunk_dir,
        )

    def _check_cache(self, url: str, skip_cache: bool) -> Optional[str]:
        """Check cache for URL content."""
        if not self.cache_enabled or skip_cache or not self.request_cache:
            return None

        cached = self.request_cache.get(url)
        if cached:
            logger.info(f"Using cached content for: {url}")
        return cached

    def _cache_response(self, url: str, content: str) -> None:
        """Cache response content."""
        if self.cache_enabled and self.request_cache:
            self.request_cache.set(url, content)

    def _convert_content(self, html_content: str, url: str, output_format: str) -> str:
        """Fallback content conversion when Rust is not available."""
        if output_format == "json":
            return self._html_to_json(html_content, url)
        if output_format == "xml":
            return self._html_to_xml(html_content, url)
        # markdown
        from markdownify import markdownify

        return markdownify(html_content, heading_style="ATX")

    def _html_to_json(self, html_content: str, base_url: str) -> str:
        """Convert HTML to JSON format."""
        soup = BeautifulSoup(html_content, "html.parser")
        document = {
            "url": base_url,
            "title": self._get_text_from_element(soup.find("title")),
            "content": [],
        }

        for element in soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "ul",
                "ol",
                "blockquote",
                "pre",
                "code",
                "a",
                "img",
            ]
        ):
            element_data = {
                "type": element.name,
                "content": self._get_text_from_element(element),
            }

            if element.name == "a" and element.get("href"):
                element_data["href"] = urljoin(base_url, element.get("href"))
            elif element.name == "img" and element.get("src"):
                element_data["src"] = urljoin(base_url, element.get("src"))
                element_data["alt"] = element.get("alt", "")

            document["content"].append(element_data)

        return json.dumps(document, indent=2, ensure_ascii=False)

    def _html_to_xml(self, html_content: str, base_url: str) -> str:
        """Convert HTML to XML format."""
        soup = BeautifulSoup(html_content, "html.parser")

        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<document>",
            f"  <url>{base_url}</url>",
        ]

        if title := soup.find("title"):
            xml_parts.append(
                f"  <title>{self._escape_xml(self._get_text_from_element(title))}</title>"
            )

        xml_parts.append("  <content>")

        for element in soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "p",
                "ul",
                "ol",
                "blockquote",
                "pre",
                "code",
                "a",
                "img",
            ]
        ):
            if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                level = element.name[1]
                xml_parts.append(
                    f'    <heading level="{level}">{self._escape_xml(self._get_text_from_element(element))}</heading>'
                )
            elif element.name == "p":
                xml_parts.append(
                    f"    <paragraph>{self._escape_xml(self._get_text_from_element(element))}</paragraph>"
                )
            elif element.name == "a" and element.get("href"):
                href = urljoin(base_url, element.get("href"))
                xml_parts.append(
                    f'    <link href="{self._escape_xml(href)}">{self._escape_xml(self._get_text_from_element(element))}</link>'
                )
            elif element.name == "img" and element.get("src"):
                src = urljoin(base_url, element.get("src"))
                alt = element.get("alt", "")
                xml_parts.append(
                    f'    <image src="{self._escape_xml(src)}" alt="{self._escape_xml(alt)}" />'
                )
            elif element.name in ["ul", "ol"]:
                list_type = "unordered" if element.name == "ul" else "ordered"
                xml_parts.append(f'    <list type="{list_type}">')
                xml_parts.extend(
                    f"      <item>{self._escape_xml(self._get_text_from_element(li))}</item>"
                    for li in element.find_all("li", recursive=False)
                )
                xml_parts.append("    </list>")
            elif element.name == "blockquote":
                xml_parts.append(
                    f"    <blockquote>{self._escape_xml(self._get_text_from_element(element))}</blockquote>"
                )
            elif element.name in ["pre", "code"]:
                xml_parts.append(
                    f"    <code>{self._escape_xml(self._get_text_from_element(element))}</code>"
                )

        xml_parts.extend(["  </content>", "</document>"])

        return "\n".join(xml_parts)

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    def _get_text_from_element(self, element: Optional[Tag]) -> str:
        """Extract clean text from a BeautifulSoup element."""
        if element is None:
            return ""
        return self._whitespace_pattern.sub(" ", element.get_text().strip())

    async def close(self):
        """Close the async HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
