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
    print("RAGnificent Basic Demo")
    print("=" * 40)
    
    # Initialize scraper
    scraper = MarkdownScraper(
        requests_per_second=1.0, 
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    # Test URL
    test_url = "https://docs.python.org/3/tutorial/introduction.html"
    print(f"Scraping: {test_url}")
    
    try:
        # Scrape content
        html_content = scraper.scrape_website(test_url)
        print(f"✓ Downloaded {len(html_content)} characters")
        
        # Convert to markdown
        markdown_content = scraper.convert_to_markdown(html_content, test_url)
        print(f"✓ Converted to {len(markdown_content)} characters of markdown")
        
        # Create chunks
        chunks = scraper.create_chunks(markdown_content, test_url)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Show sample chunk
        if chunks:
            chunk = chunks[0]
            print(f"\nSample chunk:")
            print("-" * 40)
            if hasattr(chunk, 'content'):
                content = chunk.content[:200]
                print(f"Content: {content}...")
            if hasattr(chunk, 'metadata'):
                print(f"Metadata: {chunk.metadata}")
            
        return chunks
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return []


def demo_file_scraping():
    """Demonstrate scraping from URLs file"""
    print("\nFile-based Scraping Demo")
    print("=" * 40)
    
    # Create test URLs file
    urls_content = """# Test URLs for RAG demo
https://docs.python.org/3/tutorial/introduction.html
"""
    
    urls_file = Path("test_urls.txt")
    with open(urls_file, "w") as f:
        f.write(urls_content)
    print(f"✓ Created {urls_file}")
    
    scraper = MarkdownScraper()
    
    try:
        scraped_urls = scraper.scrape_by_links_file(
            links_file=str(urls_file),
            output_dir="demo_output",
            save_chunks=True,
            chunk_dir="demo_chunks",
            chunk_format="jsonl",
        )
        
        print(f"✓ Scraped {len(scraped_urls)} URLs")
        
        # Check output
        output_dir = Path("demo_output")
        chunk_dir = Path("demo_chunks")
        
        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"✓ Created {len(files)} output files")
            
        if chunk_dir.exists():
            chunk_files = list(chunk_dir.glob("*"))
            print(f"✓ Created {len(chunk_files)} chunk files")
            
    except Exception as e:
        print(f"✗ Error: {e}")
    finally:
        # Cleanup
        if urls_file.exists():
            urls_file.unlink()


def demo_embedding_check():
    """Check if embedding functionality is available"""
    print("\nEmbedding Functionality Check")
    print("=" * 40)
    
    try:
        from RAGnificent.core.config import EmbeddingModelType
        from RAGnificent.rag.embedding import EmbeddingService
        
        print("✓ Embedding modules imported successfully")
        
        # Try to initialize embedding service
        embedding_service = EmbeddingService(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name="BAAI/bge-small-en-v1.5",
        )
        print("✓ Embedding service initialized")
        
        # Test with simple text
        test_text = "This is a test sentence for embedding."
        try:
            result = embedding_service.embed_chunk(test_text)
            if "embedding" in result:
                embedding = result["embedding"]
                print(f"✓ Generated embedding with dimension {len(embedding)}")
            else:
                print("✗ No embedding in result")
        except Exception as e:
            print(f"✗ Embedding generation failed: {e}")
            
    except ImportError as e:
        print(f"✗ Embedding modules not available: {e}")
    except Exception as e:
        print(f"✗ Embedding functionality error: {e}")


def main():
    """Run the basic demo"""
    try:
        # Test basic functionality
        chunks = demo_basic_scraping()
        
        # Test file-based scraping
        demo_file_scraping()
        
        # Check embedding functionality
        demo_embedding_check()
        
        print(f"\nBasic demo completed successfully!")
        print(f"Created {len(chunks)} chunks from web scraping")
        
    except Exception as e:
        import traceback
        print(f"Demo failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()