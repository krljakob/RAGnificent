#!/usr/bin/env python3
"""
RAGnificent Basic Demo
Simple demonstration of core scraping and chunking functionality.
"""

import sys
from pathlib import Path

# Set up environment
sys.path.insert(0, str(Path(__file__).parent.parent))

from RAGnificent.core.scraper import MarkdownScraper


def demo_basic_scraping():
    """Demonstrate basic scraping functionality"""

    # Initialize scraper
    scraper = MarkdownScraper(
        requests_per_second=1.0, chunk_size=1000, chunk_overlap=200
    )

    # Test URL
    test_url = "https://docs.python.org/3/tutorial/introduction.html"

    try:
        # Scrape content
        html_content = scraper.scrape_website(test_url)

        # Convert to markdown
        markdown_content = scraper.convert_to_markdown(html_content, test_url)

        # Create chunks
        chunks = scraper.create_chunks(markdown_content, test_url)

        # Show sample chunk
        if chunks:
            chunk = chunks[0]
            if hasattr(chunk, "content"):
                content = chunk.content[:200]
        return chunks

    except Exception as e:
        return []


def demo_file_scraping():
    """Demonstrate scraping from URLs file"""

    # Create test URLs file
    urls_content = """# Test URLs for RAG demo
https://docs.python.org/3/tutorial/introduction.html
"""

    urls_file = Path("test_urls.txt")
    with open(urls_file, "w") as f:
        f.write(urls_content)

    scraper = MarkdownScraper()

    try:
        scraped_urls = scraper.scrape_by_links_file(
            links_file=str(urls_file),
            output_dir="demo_output",
            save_chunks=True,
            chunk_dir="demo_chunks",
            chunk_format="jsonl",
        )

        # Check output
        output_dir = Path("demo_output")
        chunk_dir = Path("demo_chunks")

        if output_dir.exists():
            files = list(output_dir.glob("*"))

        if chunk_dir.exists():
            chunk_files = list(chunk_dir.glob("*"))

    except Exception as e:
        pass
    finally:
        # Cleanup
        if urls_file.exists():
            urls_file.unlink()


def demo_embedding_check():
    """Check if embedding functionality is available"""

    try:
        from RAGnificent.core.config import EmbeddingModelType
        from RAGnificent.rag.embedding import EmbeddingService

        # Try to initialize embedding service
        embedding_service = EmbeddingService(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name="BAAI/bge-small-en-v1.5",
        )

        # Test with simple text
        test_text = "This is a test sentence for embedding."
        try:
            result = embedding_service.embed_chunk(test_text)
            if "embedding" in result:
                embedding = result["embedding"]
            else:
                pass
        except Exception as e:
            pass

    except ImportError as e:
        pass
    except Exception as e:
        pass


def main():
    """Run the basic demo"""
    try:
        # Test basic functionality
        chunks = demo_basic_scraping()

        # Test file-based scraping
        demo_file_scraping()

        # Check embedding functionality
        demo_embedding_check()

    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
