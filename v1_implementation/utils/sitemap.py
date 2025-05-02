import contextlib
import xml.etree.ElementTree as ET
from typing import List
from urllib.parse import urljoin

import requests


def parse_sitemap_index(xml_content: str) -> List[str]:
    """Parse sitemap index XML and return sitemap URLs."""
    namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    try:
        root = ET.fromstring(xml_content)
        sitemaps = []
        for sitemap in root.findall('ns:sitemap', namespaces):
            loc = sitemap.find('ns:loc', namespaces)
            if loc is not None:
                sitemaps.append(loc.text)
        return sitemaps
    except ET.ParseError as e:
        raise ValueError(f"Invalid sitemap index XML: {str(e)}") from e

def parse_sitemap(xml_content: str) -> List[str]:
    """Parse sitemap XML with namespace handling and return URLs."""
    namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    try:
        root = ET.fromstring(xml_content)
        urls = []
        for url in root.findall('ns:url', namespaces):
            loc = url.find('ns:loc', namespaces)
            if loc is not None:
                urls.append(loc.text)
        return urls
    except ET.ParseError as e:
        raise ValueError(f"Invalid sitemap XML: {str(e)}") from e


def get_sitemap_urls(base_url: str) -> List[str]:
    """Discover and parse sitemap(s) for a given base URL using concurrent requests."""
    sitemap_locations = [
        'sitemap.xml',
        'sitemap_index.xml',
        'sitemap/sitemap.xml',
        'sitemap.txt',
        'sitemap/sitemap_index.xml'
    ]

    # Also check robots.txt for sitemap reference
    with contextlib.suppress(requests.RequestException):
        robots_url = urljoin(base_url, 'robots.txt')
        robots_resp = requests.get(robots_url, timeout=3)
        if robots_resp.status_code == 200:
            for line in robots_resp.text.split('\n'):
                if line.lower().startswith('sitemap:'):
                    sitemap_locations.insert(0, line.split(':', 1)[1].strip())

    with ThreadPoolExecutor(max_workers=len(sitemap_locations)) as executor:
        futures = {executor.submit(requests.get, urljoin(base_url, loc), timeout=3): loc for loc in sitemap_locations}
        for future in as_completed(futures):
            try:
                response = future.result()
                response.raise_for_status()
                if 'sitemapindex' in response.text.lower():
                    index_urls = parse_sitemap_index(response.text)
                    all_urls = []
                    with ThreadPoolExecutor(max_workers=len(index_urls)) as sub_executor:
                        sub_futures = {sub_executor.submit(requests.get, index_url, timeout=3): index_url for index_url in index_urls}
                        for sub_future in as_completed(sub_futures):
                            try:
                                sub_response = sub_future.result()
                                sub_response.raise_for_status()
                                all_urls.extend(parse_sitemap(sub_response.text))
                            except Exception:
                                continue
                    return all_urls
                return parse_sitemap(response.text)
            except Exception:
                continue
    return []
