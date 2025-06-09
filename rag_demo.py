#!/usr/bin/env python3
"""
RAGnificent Full-Stack Demo
Demonstrates end-to-end scraping, chunking, embedding, and search functionality.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Set up environment
sys.path.insert(0, str(Path(__file__).parent))

from RAGnificent.core.scraper import MarkdownScraper
from RAGnificent.rag.embedding import EmbeddingService
from RAGnificent.rag.pipeline import Pipeline
from RAGnificent.rag.search import search
from RAGnificent.rag.vector_store import VectorStore


def print_step(step: str, description: str = ""):
    """Print a step header"""
    print(f"\n{'='*60}")
    print(f"STEP: {step}")
    if description:
        print(f"INFO: {description}")
    print('='*60)


def print_results(title: str, items: List[Any], limit: int = 3):
    """Print formatted results"""
    print(f"\n{title}:")
    print("-" * 40)
    for i, item in enumerate(items[:limit]):
        try:
            if hasattr(item, 'to_dict'):
                # If it's a Chunk object with to_dict method
                item_dict = item.to_dict()
                print(f"{i+1}. {json.dumps(item_dict, indent=2)[:200]}...")
            elif isinstance(item, dict):
                print(f"{i+1}. {json.dumps(item, indent=2)[:200]}...")
            else:
                # Fallback to string representation
                print(f"{i+1}. {str(item)[:200]}...")
        except Exception as e:
            print(f"{i+1}. Error displaying item: {e}")
    if len(items) > limit:
        print(f"... and {len(items) - limit} more items")


def demo_scraping_and_chunking():
    """Demo Step 1: Scraping and Chunking"""
    print_step("1. SCRAPING & CHUNKING", "Converting web content to structured chunks")

    # URLs to test with
    test_urls = [
        "https://docs.python.org/3/tutorial/introduction.html",
        "https://docs.python.org/3/tutorial/controlflow.html"
    ]

    scraper = MarkdownScraper(
        requests_per_second=1.0,
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []

    for url in test_urls:
        print(f"\n📥 Scraping: {url}")

        # Scrape content
        html_content = scraper.scrape_website(url)
        markdown_content = scraper.convert_to_markdown(html_content, url)

        # Create chunks
        chunks = scraper.create_chunks(markdown_content, url)
        all_chunks.extend(chunks)

        print(f"✅ Generated {len(chunks)} chunks from {url}")

        # Show first few chunks
        print_results("Sample Chunks", chunks, 2)

    print(f"\n📊 Total chunks generated: {len(all_chunks)}")

    # Save chunks for next steps
    chunks_file = Path("demo_chunks.jsonl")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            if hasattr(chunk, 'to_dict'):
                chunk_dict = chunk.to_dict()
            else:
                chunk_dict = chunk
            f.write(json.dumps(chunk_dict) + '\n')

    print(f"💾 Saved chunks to: {chunks_file}")
    return all_chunks


def demo_embedding_generation(chunks: List[Dict[str, Any]]):
    """Demo Step 2: Embedding Generation"""
    print_step("2. EMBEDDING GENERATION", "Converting text to vector embeddings")

    try:
        # Initialize embedding service
        from RAGnificent.core.config import EmbeddingModelType
        embedding_service = EmbeddingService(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name="BAAI/bge-small-en-v1.5"
        )

        print(f"🤖 Using embedding service: {type(embedding_service.model).__name__}")

        # Generate embeddings for a sample of chunks
        sample_chunks = chunks[:5]  # Just first 5 for demo

        print("⚡ Generating embeddings...")
        embedded_sample = []
        for chunk in sample_chunks:
            embedded_chunk = embedding_service.embed_chunk(chunk)
            embedded_sample.append(embedded_chunk)

        embeddings = [chunk.get('embedding') for chunk in embedded_sample]

        print(f"✅ Generated {len(embeddings)} embeddings")
        if embeddings and embeddings[0] is not None:
            print(f"📊 Embedding shape: {embeddings[0].shape}")

        return embedded_sample

    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        print("ℹ️  This might be due to missing ML dependencies")
        return []


def demo_vector_storage(embedded_chunks: List[Dict[str, Any]]):
    """Demo Step 3: Vector Storage"""
    print_step("3. VECTOR STORAGE", "Storing embeddings in vector database")

    try:
        # Initialize vector store (in-memory Qdrant)
        vector_store = VectorStore(
            host=":memory:",
            collection_name="ragnificent_demo"
        )

        print(f"🗄️  Initialized vector store: {vector_store.collection_name}")

        if embedded_chunks:
            # Store embeddings
            vector_store.store_documents(embedded_chunks)
            print(f"✅ Stored {len(embedded_chunks)} documents")

            # Get collection info
            info = vector_store.get_collection_info()
            print(f"📊 Collection stats: {info}")
        else:
            print("⚠️  No embedded chunks to store")

        return vector_store

    except Exception as e:
        print(f"❌ Vector storage failed: {e}")
        print("ℹ️  This might be due to missing Qdrant dependencies")
        return None


def demo_search_functionality(vector_store, chunks: List[Dict[str, Any]]):
    """Demo Step 4: Search Functionality"""
    print_step("4. SEARCH & RETRIEVAL", "Semantic search over stored content")

    if not vector_store:
        print("⚠️  No vector store available, skipping search demo")
        return

    # Test queries
    test_queries = [
        "How do I use Python as a calculator?",
        "What are Python strings?",
        "How to define variables in Python?",
        "Python comments and syntax"
    ]

    try:
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")

            # Perform search
            results = search(
                query=query,
                collection_name="ragnificent_demo",
                limit=3,
                threshold=0.3
            )

            if results:
                print(f"📋 Found {len(results)} relevant results:")
                for i, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    content = result.get('content', '')[:150]
                    source = result.get('source_url', 'Unknown')
                    print(f"  {i}. Score: {score:.3f}")
                    print(f"     Content: {content}...")
                    print(f"     Source: {source}")
            else:
                print("❌ No results found")

    except Exception as e:
        print(f"❌ Search failed: {e}")


def demo_end_to_end_pipeline():
    """Demo Step 5: End-to-End Pipeline"""
    print_step("5. END-TO-END PIPELINE", "Complete RAG workflow using Pipeline class")

    try:
        # Initialize pipeline
        pipeline = Pipeline(
            collection_name="ragnificent_e2e",
            embedding_model="BAAI/bge-small-en-v1.5"
        )

        print("🔧 Initialized RAG pipeline")

        # Test URL
        test_url = "https://docs.python.org/3/tutorial/introduction.html"

        print(f"🚀 Running end-to-end pipeline for: {test_url}")

        if success := pipeline.process_url(test_url):
            print("✅ Pipeline completed successfully")

            # Test search
            query = "What is Python?"
            results = pipeline.search(query, limit=3)

            print(f"\n🔍 Search results for '{query}':")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.get('content', '')[:100]}...")
        else:
            print("❌ Pipeline failed")

    except Exception as e:
        print(f"❌ End-to-end pipeline failed: {e}")


def main():
    """Run the complete RAG demo"""
    print("🚀 RAGnificent Full-Stack Demo")
    print("=" * 60)
    print("This demo showcases the complete RAG pipeline:")
    print("1. Web scraping and content extraction")
    print("2. Semantic chunking for RAG")
    print("3. Text embedding generation")
    print("4. Vector storage in Qdrant")
    print("5. Semantic search and retrieval")
    print("6. End-to-end pipeline demonstration")

    start_time = time.time()

    try:
        # Step 1: Scraping and Chunking
        chunks = demo_scraping_and_chunking()

        # Step 2: Embedding Generation
        embedded_chunks = demo_embedding_generation(chunks)

        # Step 3: Vector Storage
        vector_store = demo_vector_storage(embedded_chunks)

        # Step 4: Search Functionality
        demo_search_functionality(vector_store, chunks)

        # Step 5: End-to-End Pipeline
        demo_end_to_end_pipeline()

    except KeyboardInterrupt:
        print("\n⛔ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n⏱️  Demo completed in {elapsed:.2f} seconds")
    print("\n🎉 RAGnificent Demo Complete!")
    print("\nNext steps:")
    print("- Try with your own URLs")
    print("- Experiment with different chunk sizes")
    print("- Test with production Qdrant instance")
    print("- Build your own RAG applications!")


if __name__ == "__main__":
    main()
