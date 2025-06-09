#!/usr/bin/env python3
"""
Working RAGnificent Demo
Uses the existing chunks.jsonl files that were already created by the scraper.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Set up environment
sys.path.insert(0, str(Path(__file__).parent))

def load_existing_chunks():
    """Load chunks from existing JSONL files"""
    print("📂 Loading existing chunk files...")

    chunk_files = [
        "tutorial_chunks/chunks.jsonl",
        "demo_chunks/3_tutorial_introduction/chunks.jsonl"
    ]

    all_chunks = []
    for chunk_file in chunk_files:
        if Path(chunk_file).exists():
            print(f"   📄 Loading {chunk_file}")
            with open(chunk_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        chunk = json.loads(line.strip())
                        all_chunks.append(chunk)
                    except Exception as e:
                        print(f"   ⚠️ Skipped malformed line: {e}")

    print(f"✅ Loaded {len(all_chunks)} chunks from existing files")
    return all_chunks

def generate_embeddings_for_chunks(chunks):
    """Generate embeddings for existing chunks"""
    print(f"\n🤖 Generating embeddings for {len(chunks)} chunks...")

    from RAGnificent.core.config import EmbeddingModelType
    from RAGnificent.rag.embedding import EmbeddingService

    # Initialize embedding service
    embedding_service = EmbeddingService(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
        model_name="BAAI/bge-small-en-v1.5"
    )

    embedded_chunks = []
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        try:
            # Get content from chunk
            content = chunk.get('content', '')
            if not content:
                continue

            # Generate embedding directly from content
            result = embedding_service.embed_chunk(content)

            # Create enriched chunk with embedding
            enriched_chunk = {
                'id': chunk.get('id', f'chunk_{i}'),
                'content': content,
                'embedding': result['embedding'].tolist() if hasattr(result['embedding'], 'tolist') else result['embedding'],
                'metadata': chunk.get('metadata', {}),
                'source_url': chunk.get('source_url', ''),
                'created_at': chunk.get('created_at', ''),
                'chunk_type': chunk.get('chunk_type', 'unknown')
            }

            embedded_chunks.append(enriched_chunk)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"   ⚡ Processed {i + 1}/{len(chunks)} chunks ({rate:.1f} chunks/sec)")

        except Exception as e:
            print(f"   ❌ Failed to embed chunk {i}: {e}")

    total_time = time.time() - start_time
    print(f"✅ Generated {len(embedded_chunks)} embeddings in {total_time:.2f}s")

    return embedded_chunks

def save_embedded_chunks(chunks, filename="embedded_chunks.jsonl"):
    """Save chunks with embeddings to file"""
    print(f"\n💾 Saving {len(chunks)} embedded chunks to {filename}...")

    with open(filename, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

    file_size = Path(filename).stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Saved to {filename} ({file_size:.2f} MB)")

    return filename

def search_chunks(query, embedded_chunks, top_k=3):
    """Search chunks using cosine similarity"""
    print(f"\n🔍 Searching for: '{query}'")

    from RAGnificent.core.config import EmbeddingModelType
    from RAGnificent.rag.embedding import EmbeddingService

    # Get embedding for query
    embedding_service = EmbeddingService(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
        model_name="BAAI/bge-small-en-v1.5"
    )

    query_result = embedding_service.embed_chunk(query)
    query_embedding = np.array(query_result['embedding'])

    # Calculate similarities
    similarities = []
    for chunk in embedded_chunks:
        chunk_embedding = np.array(chunk['embedding'])
        # Cosine similarity
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        similarities.append((similarity, chunk))

    # Sort by similarity and get top results
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_results = similarities[:top_k]

    print(f"📋 Found {len(top_results)} results:")
    for i, (score, chunk) in enumerate(top_results, 1):
        content_preview = chunk['content'][:150].replace('\n', ' ')
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Content: {content_preview}...")
        print(f"      Source: {chunk.get('source_url', 'Unknown')}")
        print()

    return top_results

def run_search_demo(embedded_chunks):
    """Run a series of search queries"""
    print("\n🔍 SEARCH DEMONSTRATION")
    print("=" * 50)

    test_queries = [
        "How do I use Python as a calculator?",
        "What are Python strings?",
        "How to define variables in Python?",
        "Python comments and syntax",
        "for loops in Python",
        "if statements and conditionals"
    ]

    for query in test_queries:
        search_chunks(query, embedded_chunks, top_k=2)
        print("-" * 40)

def main():
    """Main demo function"""
    print("🚀 Working RAGnificent Demo")
    print("=" * 60)
    print("This demo uses existing scraped chunks and demonstrates:")
    print("✅ Loading existing chunk data")
    print("✅ Generating embeddings")
    print("✅ Semantic search functionality")
    print("✅ End-to-end RAG workflow")

    try:
        # Step 1: Load existing chunks
        chunks = load_existing_chunks()

        if not chunks:
            print("❌ No chunks found. Please run the scraper first:")
            print("   python -m RAGnificent https://docs.python.org/3/tutorial/introduction.html -o test.md --save-chunks")
            return

        # Step 2: Generate embeddings
        embedded_chunks = generate_embeddings_for_chunks(chunks[:20])  # Limit for demo

        # Step 3: Save embedded chunks
        save_embedded_chunks(embedded_chunks)

        # Step 4: Demonstrate search
        run_search_demo(embedded_chunks)

        print("\n🎉 Working RAG Demo Complete!")
        print("\nWhat we accomplished:")
        print(f"✅ Loaded {len(chunks)} existing chunks")
        print(f"✅ Generated embeddings for {len(embedded_chunks)} chunks")
        print("✅ Demonstrated semantic search")
        print("✅ Showed complete RAG workflow")

        print("\nNext steps:")
        print("🔗 Integrate with Qdrant vector database")
        print("🌐 Build web interface")
        print("📊 Add more sophisticated ranking")
        print("⚡ Optimize for larger datasets")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
