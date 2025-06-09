#!/usr/bin/env python3
"""
Simple RAGnificent Demo
A simplified demonstration of the RAG pipeline.
"""

import sys
from pathlib import Path

# Set up environment
sys.path.insert(0, str(Path(__file__).parent))

def demo_basic_functionality():
    """Demo basic scraping and chunking functionality"""
    print("🚀 Simple RAGnificent Demo")
    print("=" * 50)

    from RAGnificent.core.scraper import MarkdownScraper

    # Initialize scraper
    scraper = MarkdownScraper(
        requests_per_second=1.0,
        chunk_size=1000,
        chunk_overlap=200
    )

    # Test URL
    test_url = "https://docs.python.org/3/tutorial/introduction.html"
    print(f"📥 Scraping: {test_url}")

    # Scrape content
    html_content = scraper.scrape_website(test_url)
    markdown_content = scraper.convert_to_markdown(html_content, test_url)

    print(f"✅ Scraped {len(markdown_content)} characters of content")
    print(f"📄 First 200 chars: {markdown_content[:200]}...")

    # Create chunks
    chunks = scraper.create_chunks(markdown_content, test_url)
    print(f"🧩 Generated {len(chunks)} chunks")

    # Show a sample chunk
    if chunks:
        chunk = chunks[0]
        print("\n📋 Sample chunk:")
        print(f"   Type: {type(chunk)}")
        if hasattr(chunk, 'content'):
            print(f"   Content: {chunk.content[:150]}...")
        if hasattr(chunk, 'metadata'):
            print(f"   Metadata keys: {list(chunk.metadata.keys()) if chunk.metadata else 'None'}")

    print("\n✅ Basic functionality working!")
    return chunks

def demo_with_urls_file():
    """Demo using the example URLs file"""
    print("\n" + "=" * 50)
    print("📂 Testing with URLs file")

    from RAGnificent.core.scraper import MarkdownScraper

    # Create a simple URLs file
    urls_content = """# Test URLs for RAG demo
https://docs.python.org/3/tutorial/introduction.html
"""

    with open("test_urls.txt", "w") as f:
        f.write(urls_content)

    scraper = MarkdownScraper()

    # Test scraping from file
    try:
        scraped_urls = scraper.scrape_by_links_file(
            links_file="test_urls.txt",
            output_dir="demo_output",
            save_chunks=True,
            chunk_dir="demo_chunks",
            chunk_format="jsonl"
        )

        print(f"✅ Successfully scraped {len(scraped_urls)} URLs")

        # Check what was created
        output_dir = Path("demo_output")
        chunk_dir = Path("demo_chunks")

        if output_dir.exists():
            files = list(output_dir.glob("*"))
            print(f"📁 Created {len(files)} output files")

        if chunk_dir.exists():
            chunk_files = list(chunk_dir.glob("*"))
            print(f"🧩 Created {len(chunk_files)} chunk files")

            # Show a sample chunk file
            for chunk_file in chunk_files:
                if chunk_file.suffix == ".jsonl":
                    with open(chunk_file, 'r') as f:
                        first_line = f.readline()
                    print(f"📋 Sample chunk: {first_line[:150]}...")
                    break

    except Exception as e:
        print(f"❌ Error: {e}")

def demo_embedding_basics():
    """Demo basic embedding functionality if available"""
    print("\n" + "=" * 50)
    print("🤖 Testing Embedding Service")

    try:
        from RAGnificent.core.config import EmbeddingModelType
        from RAGnificent.rag.embedding import EmbeddingService

        # Try to initialize embedding service
        embedding_service = EmbeddingService(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name="BAAI/bge-small-en-v1.5"
        )

        print("✅ Embedding service initialized")
        print(f"🤖 Model type: {type(embedding_service.model).__name__}")

        # Test with simple text
        test_text = "This is a test sentence for embedding."
        try:
            result = embedding_service.embed_chunk(test_text)
            print("✅ Generated embedding for test text")
            if 'embedding' in result:
                embedding = result['embedding']
                print(f"📏 Embedding dimension: {len(embedding)}")

        except Exception as e:
            print(f"⚠️ Embedding generation failed: {e}")

    except Exception as e:
        print(f"⚠️ Embedding service not available: {e}")
        print("ℹ️ This might be due to missing ML dependencies")

def main():
    """Run the simple demo"""
    try:
        # Test basic functionality
        chunks = demo_basic_functionality()

        # Test with URLs file
        demo_with_urls_file()

        # Test embedding basics
        demo_embedding_basics()

        print("\n🎉 Simple demo completed successfully!")
        print("\nNext steps:")
        print("- Install ML dependencies for full RAG: pip install sentence-transformers torch")
        print("- Try the full demo: python rag_demo.py")
        print("- Build your own RAG applications!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
