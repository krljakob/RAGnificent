"""
Example demonstrating advanced scraper customization.

This example shows how to:
1. Configure a custom scraper with specific settings
2. Use sitemap discovery with filtering
3. Process multiple URLs in parallel
4. Save both markdown content and chunks
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAGnificent.core.scraper import MarkdownScraper
from RAGnificent.utils.sitemap_utils import discover_site_urls

def main():
    """Run the custom scraper example."""
    output_dir = project_root / "data" / "custom_scraper_example"
    chunk_dir = output_dir / "chunks"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(chunk_dir, exist_ok=True)
    
    scraper = MarkdownScraper(
        requests_per_second=2.0,  # 2 requests per second
        chunk_size=1000,          # 1000 characters per chunk
        chunk_overlap=200,        # 200 characters overlap
        cache_enabled=True,       # Enable request caching
        user_agent="RAGnificent Example/1.0"  # Custom user agent
    )
    
    print("Discovering URLs from sitemap...")
    urls = discover_site_urls(
        base_url="https://docs.python.org/3/",
        min_priority=0.7,                 # Only high-priority URLs
        include_patterns=["library/*"],   # Only library documentation
        exclude_patterns=["*whatsnew*"],  # Exclude what's new pages
        limit=10                          # Limit to 10 URLs
    )
    
    print(f"Discovered {len(urls)} URLs from sitemap")
    
    print("\nScraping by sitemap...")
    scraped_urls = scraper.scrape_by_sitemap(
        base_url="https://docs.python.org/3/",
        output_dir=str(output_dir),
        min_priority=0.7,
        include_patterns=["library/functions.html"],
        limit=1,
        save_chunks=True,
        chunk_dir=str(chunk_dir),
        chunk_format="jsonl"
    )
    
    print(f"Scraped {len(scraped_urls)} URLs by sitemap")
    
    links_file = output_dir / "links.txt"
    with open(links_file, "w") as f:
        f.write("# Example links file\n")
        f.write("https://docs.python.org/3/library/functions.html\n")
        f.write("https://docs.python.org/3/library/stdtypes.html\n")
    
    print("\nScraping from links file...")
    scraper.scrape_by_links_file(
        links_file=str(links_file),
        output_dir=str(output_dir / "links_output"),
        save_chunks=True,
        output_format="markdown",
        parallel=True,
        max_workers=2
    )
    
    print("Completed scraping from links file")
    
    print("\nManual scraping and conversion...")
    url = "https://docs.python.org/3/library/functions.html"
    html_content = scraper.scrape_website(url)
    markdown_content = scraper.convert_to_markdown(html_content, url)
    
    output_file = output_dir / "manual_output.md"
    scraper.save_content(markdown_content, str(output_file))
    
    chunks = scraper.create_chunks(markdown_content, url)
    chunks_file = chunk_dir / "manual_chunks.jsonl"
    scraper.save_chunks(chunks, str(chunks_file))
    
    print(f"Saved manual output to {output_file}")
    print(f"Created {len(chunks)} chunks and saved to {chunks_file}")

if __name__ == "__main__":
    main()
