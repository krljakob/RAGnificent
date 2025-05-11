"""
Test edge cases in the search module.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

project_root = Path(__file__).parent.parent.parent
rag_path = project_root / "RAGnificent" / "rag"
sys.path.insert(0, str(project_root))

from RAGnificent.rag.search import SearchResult, SemanticSearch, get_search, search


class TestSearchEdgeCases(unittest.TestCase):
    """Test edge cases for the search module."""

    def setUp(self):
        """Set up test environment."""
        self.config_patcher = mock.patch("RAGnificent.rag.search.get_config")
        self.mock_config = self.config_patcher.start()
        self.mock_config.return_value.qdrant.collection = "test_collection"
        self.mock_config.return_value.search.enable_caching = True
        self.mock_config.return_value.search.cache_ttl = 60
        
        self.embedding_patcher = mock.patch("RAGnificent.rag.search.get_embedding_model")
        self.mock_get_embedding = self.embedding_patcher.start()
        self.mock_embedding_model = mock.MagicMock()
        self.mock_get_embedding.return_value = self.mock_embedding_model
        
        self.vector_store_patcher = mock.patch("RAGnificent.rag.search.get_vector_store")
        self.mock_get_vector_store = self.vector_store_patcher.start()
        self.mock_vector_store = mock.MagicMock()
        self.mock_get_vector_store.return_value = self.mock_vector_store
        
        self.embed_text_patcher = mock.patch("RAGnificent.rag.search.embed_text")
        self.mock_embed_text = self.embed_text_patcher.start()
        self.mock_embed_text.return_value = np.array([0.1, 0.2, 0.3])
        
        self.search = SemanticSearch()

    def tearDown(self):
        """Clean up test environment."""
        self.config_patcher.stop()
        self.embedding_patcher.stop()
        self.vector_store_patcher.stop()
        self.embed_text_patcher.stop()

    def test_search_result_from_dict(self):
        """Test SearchResult.from_dict with missing fields."""
        result = SearchResult.from_dict({})
        self.assertEqual(result.content, "", "Empty dict should create result with empty content")
        self.assertEqual(result.score, 0.0, "Empty dict should create result with zero score")
        self.assertEqual(result.document_id, "", "Empty dict should create result with empty document_id")
        
        result = SearchResult.from_dict({"content": "Test content"})
        self.assertEqual(result.content, "Test content", "Should use provided content")
        self.assertEqual(result.score, 0.0, "Should use default score when missing")
        self.assertEqual(result.metadata, {}, "Should use empty dict for metadata when missing")

    def test_search_cache_expiration(self):
        """Test search cache expiration."""
        self.mock_vector_store.search.return_value = [
            {"id": "1", "score": 0.9, "payload": {"content": "Test content"}}
        ]
        
        self.mock_config.return_value.search.cache_ttl = 0.1  # 100ms
        
        results1 = self.search.search("test query")
        self.assertEqual(len(results1), 1, "Should return 1 result")
        self.assertEqual(self.mock_vector_store.search.call_count, 1, "Should call vector store once")
        
        results2 = self.search.search("test query")
        self.assertEqual(len(results2), 1, "Should return 1 result")
        self.assertEqual(self.mock_vector_store.search.call_count, 1, "Should still have called vector store only once")
        
        time.sleep(0.2)
        
        results3 = self.search.search("test query")
        self.assertEqual(len(results3), 1, "Should return 1 result")
        self.assertEqual(self.mock_vector_store.search.call_count, 2, "Should call vector store again after cache expiry")
