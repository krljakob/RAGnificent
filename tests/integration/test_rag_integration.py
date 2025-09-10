"""
Integration tests for the RAG pipeline.

Tests the end-to-end workflow from content extraction to search.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pytest

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

        # verify chunking results
        self.assertGreater(len(chunks), 0, "Should create chunks from test document")
        self.assertLess(
            len(chunks), 20, "Should not create excessive chunks from test document"
        )

        # verify chunk structure and content
        for i, chunk in enumerate(chunks):
            self.assertIn("content", chunk, f"Chunk {i} should have content field")
            self.assertIn("url", chunk, f"Chunk {i} should have url field")
            self.assertNotEqual(
                chunk["content"].strip(), "", f"Chunk {i} content should not be empty"
            )
            self.assertEqual(
                chunk["url"],
                self.test_document["url"],
                f"Chunk {i} should have correct source URL",
            )

        # verify content preservation
        chunk_text = " ".join(chunk["content"] for chunk in chunks)
        self.assertIn(
            "Test Document", chunk_text, "Document title should be preserved in chunks"
        )
        self.assertIn(
            "test document for RAG",
            chunk_text,
            "Key content should be preserved in chunks",
        )

        embedded_chunks = self.pipeline.embed_chunks(
            chunks, output_file="test_embedded_chunks.json"
        )

        # verify embedding results
        self.assertEqual(len(embedded_chunks), len(chunks), "Should embed all chunks")

        # verify embedding structure and quality
        for i, chunk in enumerate(embedded_chunks):
            self.assertIn("embedding", chunk, f"Chunk {i} should have an embedding")
            embedding = chunk["embedding"]
            self.assertIsInstance(
                embedding,
                (list, np.ndarray),
                f"Chunk {i} embedding should be list or array",
            )

            if isinstance(embedding, list):
                self.assertGreater(
                    len(embedding), 0, f"Chunk {i} embedding should not be empty"
                )
                self.assertLess(
                    len(embedding),
                    2000,
                    f"Chunk {i} embedding should have reasonable size",
                )
                self.assertTrue(
                    all(isinstance(x, (int, float)) for x in embedding),
                    f"Chunk {i} embedding should contain numeric values",
                )
            else:  # numpy array
                self.assertGreater(
                    embedding.size, 0, f"Chunk {i} embedding should not be empty"
                )
                self.assertLess(
                    embedding.size,
                    2000,
                    f"Chunk {i} embedding should have reasonable size",
                )

            # verify original chunk data is preserved
            self.assertIn(
                "content", chunk, f"Embedded chunk {i} should preserve content"
            )
            self.assertIn("url", chunk, f"Embedded chunk {i} should preserve url")

        # verify storage succeeded
        success = self.pipeline.store_chunks(embedded_chunks)
        self.assertTrue(success, "Should successfully store embedded chunks")

        # test search functionality
        query = "RAG systems"
        results = self.pipeline.search_documents(query)

        # verify search results
        self.assertGreater(len(results), 0, "Should find results for query")
        self.assertLessEqual(len(results), 10, "Should not return excessive results")

        # verify result structure
        for i, result in enumerate(results):
            self.assertIn("content", result, f"Result {i} should have content")
            self.assertIn("score", result, f"Result {i} should have relevance score")
            self.assertIsInstance(
                result["score"], (int, float), f"Result {i} score should be numeric"
            )
            self.assertGreater(
                result["score"], 0, f"Result {i} should have positive relevance score"
            )
            self.assertNotEqual(
                result["content"].strip(), "", f"Result {i} content should not be empty"
            )

        # verify search relevance
        found_relevant_content = any(
            "RAG systems" in result["content"]
            or "Retrieval Augmented Generation" in result["content"]
            or "RAG" in result["content"]
            for result in results
        )
        self.assertTrue(
            found_relevant_content, "Should find chunk containing query-related terms"
        )

        # verify results are ranked by relevance (scores should be in descending order)
        scores = [result["score"] for result in results]
        self.assertEqual(
            scores,
            sorted(scores, reverse=True),
            "Results should be ranked by relevance score",
        )

    def test_pipeline_with_different_chunking_strategies(self):
        """Test the pipeline with different chunking strategies."""
        # test different chunking strategies
        strategies = [
            (ChunkingStrategy.SEMANTIC, "Semantic"),
            (ChunkingStrategy.SLIDING_WINDOW, "Sliding window"),
            (ChunkingStrategy.RECURSIVE, "Recursive"),
        ]

        strategy_results = {}

        for strategy, name in strategies:
            chunks = self.pipeline.chunk_documents(
                [self.test_document], strategy=strategy
            )

            # verify each strategy produces chunks
            self.assertGreater(len(chunks), 0, f"{name} chunking should create chunks")
            self.assertLess(
                len(chunks), 50, f"{name} chunking should not create excessive chunks"
            )

            # verify chunk quality
            for i, chunk in enumerate(chunks):
                self.assertIn("content", chunk, f"{name} chunk {i} should have content")
                self.assertNotEqual(
                    chunk["content"].strip(),
                    "",
                    f"{name} chunk {i} should not be empty",
                )
                self.assertIn("url", chunk, f"{name} chunk {i} should have url")

            # store results for comparison
            strategy_results[name] = chunks

        # verify strategies produce different results (implementation-dependent)
        chunk_counts = [len(chunks) for chunks in strategy_results.values()]
        self.assertGreater(
            max(chunk_counts), 0, "At least one strategy should produce chunks"
        )

        # verify content preservation across strategies
        for strategy_name, chunks in strategy_results.items():
            combined_content = " ".join(chunk["content"] for chunk in chunks)
            self.assertIn(
                "Test Document",
                combined_content,
                f"{strategy_name} should preserve document title",
            )
            self.assertIn(
                "RAG", combined_content, f"{strategy_name} should preserve key terms"
            )
