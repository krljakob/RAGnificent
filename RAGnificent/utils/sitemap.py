"""
Utility to extract URLs from a sitemap.

This module is maintained for backward compatibility and forwards all calls to sitemap_utils.py.
"""

import warnings
from typing import List

from RAGnificent.utils.sitemap_utils import get_sitemap_urls as _get_sitemap_urls

# Show a deprecation warning when importing this module
warnings.warn(
    "The sitemap.py module is deprecated and will be removed in a future version. "
    "Please use sitemap_utils.py instead.",
    DeprecationWarning,
    stacklevel=2
)

def get_sitemap_urls(base_url: str) -> List[str]:
    """
    Extract all URLs from a website's sitemap.

    This function is a wrapper around the implementation in sitemap_utils.py.

    Args:
        base_url: The base URL of the website (e.g., 'https://example.com')

    Returns:
        A list of URLs found in the sitemap
    """
    return _get_sitemap_urls(base_url)


if __name__ == "__main__":
    # Example usage
    urls = get_sitemap_urls('https://solana.com/docs')
