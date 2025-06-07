"""
Test edge cases in the pipeline module.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from RAGnificent.rag.pipeline import Pipeline
except ImportError:
    project_root = Path(__file__).parent.parent.parent
    rag_path = project_root / "RAGnificent" / "rag"
    sys.path.insert(0, str(project_root))
    from RAGnificent.rag.pipeline import Pipeline


class TestPipelineEdgeCases(unittest.TestCase):
    """Test edge cases for the RAG pipeline."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        self.pipeline = Pipeline(
            collection_name="test_collection", data_dir=self.data_dir
        )

        self.test_files_dir = self.data_dir / "test_files"
        os.makedirs(self.test_files_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_process_invalid_url(self):
        """Test handling of invalid URLs."""
        result = self.pipeline._process_single_url("not_a_url", "markdown", False)
        self.assertIsNone(result, "Invalid URL should return None")

        result = self.pipeline._process_single_url("http://", "markdown", False)
        self.assertIsNone(result, "Malformed URL should return None")

    @mock.patch("RAGnificent.core.scraper.MarkdownScraper.scrape_website")
    def test_network_error_handling(self, mock_scrape):
        """Test handling of network errors during scraping."""
        mock_scrape.side_effect = Exception("Network error")

        result = self.pipeline._process_single_url(
            "https://example.com", "markdown", False
        )
        self.assertIsNone(result, "Network error should return None")

    def test_empty_document_list(self):
        """Test handling of empty document list."""
        chunks = self.pipeline.chunk_documents([])
        self.assertEqual(len(chunks), 0, "Empty document list should yield no chunks")

    def test_malformed_document(self):
        """Test handling of malformed documents."""
        malformed_docs = [
            {"url": "https://example.com"},  # Missing content
            {"content": "Some content"},  # Missing URL
            {},  # Empty document
        ]

        chunks = self.pipeline.chunk_documents(malformed_docs)
        self.assertEqual(len(chunks), 0, "Malformed documents should yield no chunks")

    def test_invalid_chunking_strategy(self):
        """Test handling of invalid chunking strategy."""
        doc = {
            "url": "https://example.com",
            "content": "# Test\nSome content for testing.",
            "title": "Test Document",
        }

        with mock.patch("RAGnificent.core.config.get_config") as mock_config:
            mock_config.return_value.chunking.strategy = "invalid_strategy"
            chunks = self.pipeline.chunk_documents([doc])
            self.assertEqual(
                len(chunks), 0, "Invalid chunking strategy should yield no chunks"
            )

    @mock.patch("RAGnificent.rag.embedding.get_embedding_service")
    def test_embedding_failure(self, mock_get_embedding):
        """Test handling of embedding failures."""
        mock_embedding_service = mock.MagicMock()
        mock_embedding_service.embed_chunks.side_effect = Exception("Embedding failed")
        mock_get_embedding.return_value = mock_embedding_service

        pipeline = Pipeline(data_dir=self.data_dir)

        chunks = [
            {"id": "1", "content": "Test content 1"},
            {"id": "2", "content": "Test content 2"},
        ]

        result = pipeline.embed_chunks(chunks)
        self.assertEqual(
            result,
            chunks,
            "Should return original chunks without embeddings on failure",
        )

    def test_empty_chunks_embedding(self):
        """Test handling of empty chunks for embedding."""
        result = self.pipeline.embed_chunks([])
        self.assertEqual(len(result), 0, "Empty chunks list should yield no embeddings")

    def test_store_chunks_without_embeddings(self):
        """Test handling of storing chunks without embeddings."""
        chunks = [
            {"id": "1", "content": "Test content 1"},
            {"id": "2", "content": "Test content 2"},
        ]

        result = self.pipeline.store_chunks(chunks)
        self.assertFalse(
            result, "Storing chunks without embeddings should return False"
        )
