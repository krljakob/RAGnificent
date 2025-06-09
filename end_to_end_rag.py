#!/usr/bin/env python3
"""
RAGnificent End-to-End Demo
Complete demonstration of scraping, chunking, embedding, and search.
"""

import json
import sys
import time
from pathlib import Path

# Set up environment
sys.path.insert(0, str(Path(__file__).parent))

def run_end_to_end_workflow():
    """Run a complete end-to-end RAG workflow"""
    print("🚀 RAGnificent End-to-End RAG Workflow")
    print("=" * 60)

    # Step 1: Setup and Configuration
    print("\n📋 STEP 1: Setup and Configuration")
    print("-" * 40)

    from RAGnificent.core.config import EmbeddingModelType
    from RAGnificent.core.scraper import MarkdownScraper
    from RAGnificent.rag.embedding import EmbeddingService

    # Test URLs for the demo
    demo_urls = [
        "https://docs.python.org/3/tutorial/introduction.html",
        "https://docs.python.org/3/tutorial/controlflow.html",
    ]

    print(f"📑 Target URLs: {len(demo_urls)}")
    for i, url in enumerate(demo_urls, 1):
        print(f"   {i}. {url}")

    # Initialize components
    scraper = MarkdownScraper(
        requests_per_second=1.0,
        chunk_size=1000,
        chunk_overlap=200
    )

    embedding_service = EmbeddingService(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
        model_name="BAAI/bge-small-en-v1.5"
    )

    print("✅ Components initialized")

    # Step 2: Content Extraction and Chunking
    print("\n📥 STEP 2: Content Extraction and Chunking")
    print("-" * 40)

    all_chunks = []
    all_embedded_chunks = []

    for i, url in enumerate(demo_urls, 1):
        print(f"\n🌐 Processing URL {i}/{len(demo_urls)}: {url}")

        # Scrape content
        start_time = time.time()
        html_content = scraper.scrape_website(url)
        markdown_content = scraper.convert_to_markdown(html_content, url)
        scrape_time = time.time() - start_time

        print(f"   📄 Scraped {len(markdown_content):,} characters in {scrape_time:.2f}s")

        # Create chunks
        start_time = time.time()
        chunks = scraper.create_chunks(markdown_content, url)
        chunk_time = time.time() - start_time

        print(f"   🧩 Generated {len(chunks)} chunks in {chunk_time:.2f}s")
        all_chunks.extend(chunks)

        # Generate embeddings for each chunk
        print("   🤖 Generating embeddings...")
        start_time = time.time()

        embedded_chunks_for_url = []
        for j, chunk in enumerate(chunks):
            try:
                embedded_chunk = embedding_service.embed_chunk(chunk)
                embedded_chunks_for_url.append(embedded_chunk)

                if (j + 1) % 10 == 0:
                    print(f"      ⚡ Processed {j + 1}/{len(chunks)} chunks")

            except Exception as e:
                print(f"      ❌ Failed to embed chunk {j + 1}: {e}")

        embedding_time = time.time() - start_time
        print(f"   ✅ Generated {len(embedded_chunks_for_url)} embeddings in {embedding_time:.2f}s")

        all_embedded_chunks.extend(embedded_chunks_for_url)

    print("\n📊 TOTALS:")
    print(f"   🧩 Total chunks: {len(all_chunks)}")
    print(f"   🤖 Total embeddings: {len(all_embedded_chunks)}")

    # Step 3: Save Processed Data
    print("\n💾 STEP 3: Save Processed Data")
    print("-" * 40)

    # Save chunks with embeddings
    output_file = Path("rag_demo_data.jsonl")
    saved_count = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_embedded_chunks:
            try:
                # Convert numpy arrays to lists for JSON serialization
                chunk_data = {
                    'content': chunk.get('content', ''),
                    'embedding': chunk.get('embedding', []),
                    'metadata': getattr(chunk, 'metadata', {}) if hasattr(chunk, 'metadata') else {},
                    'source_url': getattr(chunk, 'source_url', '') if hasattr(chunk, 'source_url') else '',
                    'chunk_id': getattr(chunk, 'id', '') if hasattr(chunk, 'id') else f"chunk_{saved_count}"
                }

                # Ensure embedding is a list
                if hasattr(chunk_data['embedding'], 'tolist'):
                    chunk_data['embedding'] = chunk_data['embedding'].tolist()

                f.write(json.dumps(chunk_data) + '\n')
                saved_count += 1

            except Exception as e:
                print(f"⚠️ Failed to save chunk: {e}")

    print(f"✅ Saved {saved_count} chunks with embeddings to {output_file}")

    # Step 4: Demonstrate Search Functionality
    print("\n🔍 STEP 4: Search Demonstration")
    print("-" * 40)

    # Simple search function using cosine similarity
    def simple_search(query_text, chunks_data, top_k=3):
        """Simple cosine similarity search"""
        import numpy as np

        # Get query embedding
        query_result = embedding_service.embed_chunk(query_text)
        query_embedding = np.array(query_result['embedding'])

        # Calculate similarities
        similarities = []
        for i, chunk in enumerate(chunks_data):
            if chunk.get('embedding'):
                chunk_embedding = np.array(chunk['embedding'])
                # Cosine similarity
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append((similarity, i, chunk))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]

    # Load saved data for search
    search_data = []
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            search_data.append(json.loads(line.strip()))

    print(f"📚 Loaded {len(search_data)} chunks for search")

    # Test queries
    test_queries = [
        "How do I use Python as a calculator?",
        "What are Python strings and how do I use them?",
        "Explain Python variables and assignment",
        "How do if statements work in Python?"
    ]

    print(f"\n🔍 Testing {len(test_queries)} search queries:")

    for query in test_queries:
        print(f"\n❓ Query: '{query}'")

        try:
            start_time = time.time()
            results = simple_search(query, search_data, top_k=3)
            search_time = time.time() - start_time

            print(f"   ⚡ Found {len(results)} results in {search_time:.3f}s")

            for i, (similarity, idx, chunk) in enumerate(results, 1):
                content_preview = chunk['content'][:150].replace('\n', ' ')
                print(f"   {i}. Score: {similarity:.3f}")
                print(f"      Content: {content_preview}...")
                print(f"      Source: {chunk.get('source_url', 'Unknown')}")

        except Exception as e:
            print(f"   ❌ Search failed: {e}")

    # Step 5: Performance Summary
    print("\n📈 STEP 5: Performance Summary")
    print("-" * 40)

    print(f"✅ Successfully processed {len(demo_urls)} URLs")
    print(f"🧩 Generated {len(all_chunks)} total chunks")
    print(f"🤖 Created {len(all_embedded_chunks)} embeddings")
    print(f"💾 Saved {saved_count} chunks to {output_file}")
    print("🔍 Demonstrated semantic search functionality")

    # File size info
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        print(f"📁 Output file size: {file_size:.2f} MB")

    print("\n🎉 End-to-End RAG Workflow Complete!")
    print("\nWhat this demo accomplished:")
    print("✅ Web scraping with rate limiting")
    print("✅ Semantic chunking with metadata")
    print("✅ Text embedding generation")
    print("✅ Data persistence in JSONL format")
    print("✅ Semantic search with cosine similarity")
    print("✅ Performance monitoring throughout")

    print("\nNext steps for production:")
    print("🔗 Integrate with vector database (Qdrant/Pinecone)")
    print("🎯 Add more sophisticated reranking")
    print("🌐 Build web API endpoints")
    print("📊 Add monitoring and analytics")
    print("🔧 Optimize for larger datasets")

def main():
    """Main entry point"""
    try:
        run_end_to_end_workflow()
    except KeyboardInterrupt:
        print("\n⛔ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
