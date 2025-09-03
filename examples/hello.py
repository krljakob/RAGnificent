#!/usr/bin/env python3
"""
Simple hello world example for RAGnificent.

This demonstrates basic usage of the MarkdownScraper.
"""

import logging

from RAGnificent.core.scraper import MarkdownScraper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run a simple scraping example."""

    # Create scraper instance
    scraper = MarkdownScraper()

    # Example URL - Python's official website
    url = "https://www.python.org"

    try:
        html_content = scraper.scrape_website(url)

        try:
            from ragnificent_rs import OutputFormat

            markdown_content = scraper.convert_html(
                html_content, url, OutputFormat.MARKDOWN
            )
        except (ImportError, AttributeError):
            # Fallback for different enum structure
            markdown_content = scraper.convert_html(html_content, url, "markdown")

        # Display first 500 characters
        preview = (
            f"{markdown_content[:500]}..."
            if len(markdown_content) > 500
            else markdown_content
        )

    except Exception as e:
        logger.error(f"Error in hello example: {e}")


if __name__ == "__main__":
    main()
