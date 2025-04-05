"""
Utility to extract URLs from a sitemap.
"""
import re
import requests
from typing import List
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urljoin


def get_sitemap_urls(base_url: str) -> List[str]:
    """
    Extract all URLs from a website's sitemap.
    
    Args:
        base_url: The base URL of the website (e.g., 'https://example.com')
        
    Returns:
        A list of URLs found in the sitemap
    """
    # First try to locate the sitemap at the standard location
    sitemap_url = urljoin(base_url, 'sitemap.xml')
    response = requests.get(sitemap_url)
    
    # If not found at standard location, try robots.txt
    if response.status_code != 200:
        robots_url = urljoin(base_url, 'robots.txt')
        robots_response = requests.get(robots_url)
        
        if robots_response.status_code == 200:
            # Look for sitemap declarations in robots.txt
            sitemap_matches = re.findall(r'Sitemap: (.*)', robots_response.text, re.IGNORECASE)
            if sitemap_matches:
                sitemap_url = sitemap_matches[0].strip()
                response = requests.get(sitemap_url)
    
    urls = []
    
    if response.status_code == 200:
        content_type = response.headers.get('Content-Type', '')
        
        # Handle XML sitemaps
        if 'xml' in content_type:
            try:
                # Parse XML
                root = ET.fromstring(response.content)
                
                # Define namespace map if needed
                ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                
                # Extract URLs from sitemap using list comprehension instead of for-append loop
                urls.extend(
                    url_elem.text.strip() 
                    for url_elem in (root.findall('.//sm:url/sm:loc', ns) or root.findall('.//url/loc'))
                    if url_elem is not None and url_elem.text
                )
                
                # Check if this is a sitemap index that points to other sitemaps
                for sitemap_elem in root.findall('.//sm:sitemap/sm:loc', ns) or root.findall('.//sitemap/loc'):
                    if sitemap_elem is not None and sitemap_elem.text:
                        sub_sitemap_url = sitemap_elem.text.strip()
                        sub_response = requests.get(sub_sitemap_url)
                        if sub_response.status_code == 200:
                            sub_root = ET.fromstring(sub_response.content)
                            # Use list extend instead of for-append loop
                            urls.extend(
                                url_elem.text.strip()
                                for url_elem in (sub_root.findall('.//sm:url/sm:loc', ns) or sub_root.findall('.//url/loc'))
                                if url_elem is not None and url_elem.text
                            )
            except ET.ParseError:
                # If XML parsing fails, try HTML parsing
                soup = BeautifulSoup(response.content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    urls.append(urljoin(sitemap_url, link['href']))
        
        # Handle HTML sitemaps
        elif 'html' in content_type:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Use list extend instead of for-append loop
            urls.extend(urljoin(sitemap_url, link['href']) for link in soup.find_all('a', href=True))
    
    # Filter to only include URLs from the target domain
    domain = re.search(r'https?://([^/]+)', base_url).group(1)
    filtered_urls = [url for url in urls if domain in url]
    
    # Filter specifically for docs pages if looking at docs
    if '/docs' in base_url:
        filtered_urls = [url for url in filtered_urls if '/docs' in url]
    
    return filtered_urls


if __name__ == "__main__":
    # Example usage
    urls = get_sitemap_urls('https://solana.com/docs')
    print(f"Found {len(urls)} URLs in the sitemap")
    for url in urls[:5]:  # Print first 5 URLs as a sample
        print(url)
