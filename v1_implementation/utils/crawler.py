#!/usr/bin/env python
"""Web crawler utility for RAG implementation

Provides fallback crawling when sitemap is not available.
"""
import logging
from typing import List
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def is_valid_url(url: str, base_domain: str) -> bool:
    """Check if URL is valid and belongs to same domain."""
    parsed = urlparse(url)
    return (parsed.scheme in ('http', 'https') and 
           (parsed.netloc == base_domain or parsed.netloc.endswith(f'.{base_domain}')))


def crawl_urls(start_url: str, max_pages: int = 20) -> List[str]:
    """Crawl website starting from given URL.
    
    Args:
        start_url: URL to start crawling from
        max_pages: Maximum number of pages to crawl
        
    Returns:
        List of unique URLs found
    """
    base_domain = urlparse(start_url).netloc
    visited = set()
    queue = [start_url]
    urls = []
    
    try:
        while queue and len(urls) < max_pages:
            url = queue.pop(0)
            
            if url in visited:
                continue
                
            visited.add(url)
            
            try:
                # Fetch page content
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                urls.append(url)
                logger.info(f"Crawled: {url}")
                
                # Extract and queue valid links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(url, href)
                    if is_valid_url(absolute_url, base_domain) and absolute_url not in visited:
                        queue.append(absolute_url)
                        
            except Exception as e:
                logger.warning(f"Failed to crawl {url}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Crawling failed: {str(e)}")
    
    return urls


if __name__ == "__main__":
    # Test crawling
    import sys
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
        print(f"Crawling {test_url}...")
        results = crawl_urls(test_url)
        print(f"Found {len(results)} URLs:")
        for url in results:
            print(f"- {url}")
    else:
        print("Please provide a URL to crawl")
