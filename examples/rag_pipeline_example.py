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

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAGnificent.core.config import ChunkingStrategy, EmbeddingModelType
from RAGnificent.rag.pipeline import Pipeline


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
        data_dir=project_root / "data" / "example",
    )

    documents = pipeline.extract_content(
        url="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        output_file="extracted_documents.json",
        output_format="markdown",
        limit=5,  # Limit to 5 pages if using sitemap
    )

    chunks = pipeline.chunk_documents(
        documents=documents,
        output_file="document_chunks.json",
        strategy=ChunkingStrategy.SEMANTIC,
    )

    embedded_chunks = pipeline.embed_chunks(
        chunks=chunks, output_file="embedded_chunks.json"
    )

    if success := pipeline.store_chunks(embedded_chunks):
        search_results = pipeline.search_documents(
            query="What is retrieval-augmented generation?", limit=3
        )

        pipeline.query_with_context(
            query="Explain retrieval-augmented generation in simple terms",
            max_tokens=200,
        )


if __name__ == "__main__":
    main()
