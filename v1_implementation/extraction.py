#!/usr/bin/env python
"""Document extraction module for RAG implementation

This script extracts content from web pages and converts them to a standardized
document format that can be used in the RAG pipeline. It includes robust error
handling, input validation, rate limiting, and structured logging.
"""
import json
import logging
import os
import re
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field, HttpUrl, ValidationError, validator

from config import ExtractionConfig, load_config
from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
data_dir = Path(__file__).parent.parent / 'data'
os.makedirs(data_dir, exist_ok=True)


# Define data models for validation
class ExtractionOptions(BaseModel):
    """Options for document extraction"""
    base_url: HttpUrl = Field(..., description="Base URL to extract documents from")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of URLs to process")
    debug: bool = Field(False, description="Enable debug logging")
    output_path: Optional[Path] = Field(None, description="Custom output path")
    timeout: int = Field(10, ge=1, le=60, description="Request timeout in seconds")
    rate_limit: float = Field(0.5, ge=0.1, description="Seconds between requests")
    use_sitemap: bool = Field(True, description="Whether to use sitemap for URL discovery")
    follow_links: bool = Field(True, description="Whether to follow links during extraction")
    max_depth: int = Field(2, ge=0, le=5, description="Maximum depth for link following")
    user_agent: str = Field(
        "Mozilla/5.0 RAGnificent/1.0",
        min_length=5,
        description="User agent for web requests"
    )
    respect_robots_txt: bool = Field(True, description="Whether to respect robots.txt")

    @validator('base_url')
    def validate_url(cls, v):
        """Additional URL validation beyond pydantic's HttpUrl"""
        # Convert HttpUrl to string before parsing
        url_str = str(v)
        parsed = urlparse(url_str)
        if not parsed.netloc or not parsed.scheme:
            raise ValueError(f"Invalid URL format: {url_str}")
        return url_str  # Return as string


def rate_limited(max_per_second: float):
    """Decorator to rate limit function calls
    
    Args:
        max_per_second: Maximum calls per second
    """
    min_interval = 1.0 / max_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            to_wait = min_interval - elapsed
            if to_wait > 0:
                logger.debug(f"Rate limiting: sleeping for {to_wait:.2f}s")
                time.sleep(to_wait)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_and_normalize_url(url: str) -> Optional[str]:
    """Validate URL format and normalize
    
    Args:
        url: URL to validate and normalize
        
    Returns:
        Normalized URL or None if invalid
    """
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            logger.warning(f"Invalid URL format: {url}")
            return None
            
        # Normalize URL (remove fragments, trailing slashes)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized
    except Exception as e:
        logger.warning(f"URL validation error for {url}: {str(e)}")
        return None


@rate_limited(2.0)  # Maximum 2 requests per second
def fetch_url(url: str, timeout: int = 10, user_agent: str = "RAGnificent/1.0") -> Optional[str]:
    """Fetch URL content with error handling and timeouts
    
    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        user_agent: User agent string for request
        
    Returns:
        URL content or None if fetch failed
    """
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        logger.debug(f"Fetching URL: {url}")
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching {url}: {e.response.status_code} {e.response.reason}")
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error fetching {url}")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching {url} after {timeout}s")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {url}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error fetching {url}: {str(e)}")
    
    return None


def extract_documents(
    base_url: Union[str, HttpUrl],
    limit: Optional[int] = None,
    debug: bool = False,
    config: Optional[ExtractionConfig] = None,
    output_path: Optional[Union[str, Path]] = None
) -> List[Dict]:
    """Extract documents from a website with improved validation and error handling.

    Args:
        base_url: The base URL to extract documents from
        limit: Maximum number of URLs to process (None for all)
        debug: Enable debug logging
        config: Optional extraction configuration
        output_path: Custom output path for extracted documents

    Returns:
        A list of extracted documents

    Raises:
        ValueError: If inputs are invalid
        IOError: If document saving fails
    """
    # Validate inputs using pydantic model
    try:
        options = ExtractionOptions(
            base_url=base_url,
            limit=limit,
            debug=debug,
            output_path=output_path,
        )
    except ValidationError as e:
        logger.error(f"Validation error in extraction parameters: {e}")
        raise ValueError(f"Invalid extraction parameters: {e}") from e

    # Set logging level based on debug flag
    if options.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Load configuration if not provided
    if config is None:
        app_config = load_config()
        config = app_config.extraction
        logger.debug("Using default extraction configuration")

    # Update options from config
    options.timeout = config.timeout
    options.rate_limit = config.rate_limit
    options.use_sitemap = config.use_sitemap
    options.follow_links = config.follow_links
    options.max_depth = config.max_depth
    options.user_agent = config.user_agent
    options.respect_robots_txt = config.respect_robots_txt

    logger.info(f"Extracting documents from {options.base_url}")
    start_time = time.time()

    # Get URLs from sitemap with fallback to crawling
    urls = []
    if options.use_sitemap:
        try:
            # Convert HttpUrl to string before passing to the function
            base_url_str = str(options.base_url)
            urls = get_sitemap_urls(base_url_str)
            logger.info(f"Found {len(urls)} URLs in sitemap")
        except Exception as e:
            logger.warning(f"Sitemap discovery failed: {str(e)}")
            if options.follow_links:
                logger.info("Falling back to crawling base URL and linked pages")

    # If no URLs from sitemap and following links is enabled, try crawling
    if not urls and options.follow_links:
        try:
            from utils.crawler import crawl_urls
            # Convert HttpUrl to string before passing to the function
            base_url_str = str(options.base_url)
            # Only pass parameters that the crawler function accepts
            urls = crawl_urls(
                base_url_str,
                max_pages=options.limit or 100
            )
            logger.info(f"Crawled {len(urls)} URLs")
        except ImportError:
            logger.error("Crawler module not available")
            # Convert HttpUrl to string
            base_url_str = str(options.base_url)
            urls = [base_url_str]
        except Exception as e:
            logger.error(f"Crawling failed: {str(e)}")
            # Convert HttpUrl to string
            base_url_str = str(options.base_url)
            urls = [base_url_str]

    # If still no URLs, use base URL only
    if not urls:
        logger.warning("No URLs found from sitemap or crawling. Using base URL only.")
        # Convert HttpUrl to string
        base_url_str = str(options.base_url)
        urls = [base_url_str]

    # Validate and normalize all URLs
    validated_urls = []
    for url in urls:
        normalized = validate_and_normalize_url(url)
        if normalized and normalized not in validated_urls:
            validated_urls.append(normalized)

    urls = validated_urls
    logger.info(f"Found {len(urls)} valid unique URLs to process")

    # Apply limit if specified
    if options.limit and len(urls) > options.limit:
        urls = urls[:options.limit]
        logger.info(f"Limited to processing {options.limit} URLs (from {len(validated_urls)} available)")

    # Initialize converter with proper error handling
    try:
        converter = DocumentConverter()
    except Exception as e:
        logger.error(f"Failed to initialize document converter: {str(e)}")
        raise RuntimeError(f"Document converter initialization failed: {str(e)}") from e

    # Process each URL with proper progress tracking and error handling
    successful = 0
    failed = 0
    skipped = 0

    for i, url in enumerate(urls):
        try:
            logger.info(f"Processing [{i+1}/{len(urls)}]: {url}")
            
            # Check if the URL is still valid
            if not validate_and_normalize_url(url):
                logger.warning(f"Skipping invalid URL: {url}")
                skipped += 1
                continue
                
            # Apply rate limiting
            if i > 0:
                time.sleep(options.rate_limit)
                
            # Convert the document
            success = converter.convert(url)
            
            if success:
                successful += 1
                logger.debug(f"Successfully processed {url}")
            else:
                logger.warning(f"Conversion returned False for {url}")
                failed += 1
                
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            failed += 1

    # Calculate statistics
    elapsed = time.time() - start_time
    success_rate = successful / max(1, len(urls)) * 100
    
    logger.info(f"Extraction completed in {elapsed:.2f}s")
    logger.info(f"Successfully processed {successful} out of {len(urls)} URLs ({success_rate:.1f}%)")
    logger.info(f"Failed: {failed}, Skipped: {skipped}")

    # Determine output path
    if options.output_path:
        output_path = Path(options.output_path)
    else:
        # Use default path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = data_dir / f'raw_documents_{timestamp}.json'

    # Ensure output directory exists
    os.makedirs(output_path.parent, exist_ok=True)

    # Save to disk with error handling
    try:
        converter.save(str(output_path))
        logger.info(f"Saved raw documents to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save documents: {str(e)}")
        raise IOError(f"Failed to save documents: {str(e)}") from e

    # Return the documents with validation
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                documents = json.load(f)
                logger.info(f"Loaded {len(documents)} documents from {output_path}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in documents file: {str(e)}")
                documents = []
    except IOError as e:
        logger.error(f"Failed to read documents file: {str(e)}")
        documents = []

    if not documents:
        logger.warning("No documents were extracted or loaded")

    return documents


if __name__ == "__main__":
    # Configure logging for CLI usage
    logging.basicConfig(level=logging.INFO)
    
    # Extract documents
    base_url = "https://solana.com/docs"
    
    try:
        documents = extract_documents(base_url, limit=20, debug=True)  # Limit for testing
        print(f"Extracted {len(documents)} documents")
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        print(f"Error: {str(e)}")
        exit(1)
