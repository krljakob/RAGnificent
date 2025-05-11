"""
Example demonstrating the complete RAG pipeline usage.

This example shows how to:
1. Extract content from websites
2. Chunk the content semantically
3. Generate embeddings for the chunks
4. Store the embeddings in a vector database
5. Perform semantic search
6. Generate responses using retrieved context
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAGnificent.rag.pipeline import Pipeline
from RAGnificent.core.config import ChunkingStrategy, EmbeddingModelType, load_config


def main():
    """Run the complete RAG pipeline example."""
    pipeline = Pipeline(
        collection_name="example_collection",
        embedding_model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
        embedding_model_name="BAAI/bge-small-en-v1.5",
        chunk_size=800,
        chunk_overlap=150,
        requests_per_second=2.0,
        cache_enabled=True,
        data_dir=project_root / "data" / "example"
    )

    print("1. Extracting content from websites...")
    documents = pipeline.extract_content(
        url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        output_file="extracted_documents.json",
        output_format="markdown",
        limit=5  # Limit to 5 pages if using sitemap
    )

    print(f"Extracted {len(documents)} documents")

    print("\n2. Chunking documents...")
    chunks = pipeline.chunk_documents(
        documents=documents,
        output_file="document_chunks.json",
        strategy=ChunkingStrategy.SEMANTIC
    )

    print(f"Created {len(chunks)} chunks")

    print("\n3. Generating embeddings...")
    embedded_chunks = pipeline.embed_chunks(
        chunks=chunks,
        output_file="embedded_chunks.json"
    )

    print(f"Generated embeddings for {len(embedded_chunks)} chunks")

    print("\n4. Storing chunks in vector database...")
    if success := pipeline.store_chunks(embedded_chunks):
        print("Successfully stored chunks in vector database")

        print("\n5. Performing semantic search...")
        search_results = pipeline.search_documents(
            query="What is retrieval-augmented generation?",
            limit=3
        )

        print(f"Found {len(search_results)} relevant chunks")
        for i, result in enumerate(search_results):
            print(f"\nResult {i+1} (Score: {result.score:.4f}):")
            print(f"Content: {result.content[:150]}...")

        print("\n6. Generating response with context...")
        response = pipeline.query_with_context(
            query="Explain retrieval-augmented generation in simple terms",
            max_tokens=200
        )

        print("\nGenerated Response:")
        print(response)
    else:
        print("Failed to store chunks in vector database")

if __name__ == "__main__":
    main()
