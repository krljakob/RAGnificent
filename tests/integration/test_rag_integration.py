"""
Integration tests for the RAG pipeline.

Tests the end-to-end workflow from content extraction to search.
"""

import pytest
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from RAGnificent.core.config import ChunkingStrategy
    from RAGnificent.rag.pipeline import Pipeline
except ImportError:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from RAGnificent.core.config import ChunkingStrategy
    from RAGnificent.rag.pipeline import Pipeline


@pytest.mark.integration
@pytest.mark.requires_model
class TestRAGIntegration(unittest.TestCase):
    """Integration tests for the RAG pipeline."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        self.pipeline = Pipeline(
            collection_name="test_integration", data_dir=self.data_dir
        )

        self.test_content = """# Test Document

This is a test document for RAG integration testing.


This is the first section of the test document.


This is a subsection with some specific content about RAG systems.
Retrieval Augmented Generation combines search with generation.


This is the second section of the test document.


This subsection contains information about testing methodologies.
Integration testing verifies that components work together correctly.
"""

        self.test_document = {
            "url": "https://example.com/test",
            "content": self.test_content,
            "title": "Test Document",
            "format": "markdown",
            "timestamp": "2023-01-01T00:00:00",
        }

        self.documents_file = self.data_dir / "test_documents.json"
        self.pipeline._save_documents([self.test_document], "test_documents.json")

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_end_to_end_pipeline(self):
        """Test the complete RAG pipeline end-to-end."""
        chunks = self.pipeline.chunk_documents(
            [self.test_document],
            output_file="test_chunks.json",
            strategy=ChunkingStrategy.SEMANTIC,
        )

        self.assertGreater(len(chunks), 0, "Should create chunks from test document")

        embedded_chunks = self.pipeline.embed_chunks(
            chunks, output_file="test_embedded_chunks.json"
        )

        self.assertEqual(len(embedded_chunks), len(chunks), "Should embed all chunks")
        for chunk in embedded_chunks:
            self.assertIn("embedding", chunk, "Each chunk should have an embedding")

        if success := self.pipeline.store_chunks(embedded_chunks):
            results = self.pipeline.search_documents("RAG systems")

            self.assertGreater(len(results), 0, "Should find results for query")

            found_relevant_content = any(
                "RAG systems" in result["content"]
                or "Retrieval Augmented Generation" in result["content"]
                for result in results
            )
            self.assertTrue(
                found_relevant_content, "Should find chunk containing query terms"
            )

    def test_pipeline_with_different_chunking_strategies(self):
        """Test the pipeline with different chunking strategies."""
        semantic_chunks = self.pipeline.chunk_documents(
            [self.test_document], strategy=ChunkingStrategy.SEMANTIC
        )

        sliding_chunks = self.pipeline.chunk_documents(
            [self.test_document], strategy=ChunkingStrategy.SLIDING_WINDOW
        )

        recursive_chunks = self.pipeline.chunk_documents(
            [self.test_document], strategy=ChunkingStrategy.RECURSIVE
        )

        self.assertGreater(
            len(semantic_chunks), 0, "Semantic chunking should create chunks"
        )
        self.assertGreater(
            len(sliding_chunks), 0, "Sliding window chunking should create chunks"
        )
        self.assertGreater(
            len(recursive_chunks), 0, "Recursive chunking should create chunks"
        )
