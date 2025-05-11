"""
Test edge cases in the search module (part 2).
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

project_root = Path(__file__).parent.parent.parent
rag_path = project_root / "RAGnificent" / "rag"
sys.path.insert(0, str(project_root))

from RAGnificent.rag.search import SearchResult, SemanticSearch, get_search, search


class TestSearchEdgeCasesPart2(unittest.TestCase):
    """Test edge cases for the search module (part 2)."""

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

    def test_embedding_error_handling(self):
        """Test handling of embedding errors during search."""
        self.mock_embed_text.side_effect = Exception("Embedding error")
        
        results = self.search.search("test query")
        self.assertEqual(len(results), 0, "Should return empty results on embedding error")

    def test_vector_store_error_handling(self):
        """Test handling of vector store errors during search."""
        self.mock_vector_store.search.side_effect = Exception("Vector store error")
        
        results = self.search.search("test query")
        self.assertEqual(len(results), 0, "Should return empty results on vector store error")

    def test_empty_search_results(self):
        """Test handling of empty search results."""
        self.mock_vector_store.search.return_value = []
        
        results = self.search.search("test query")
        self.assertEqual(len(results), 0, "Should return empty results when vector store returns none")

    def test_malformed_search_results(self):
        """Test handling of malformed search results from vector store."""
        self.mock_vector_store.search.return_value = [
            {},  # Missing id and payload
            {"id": "1"},  # Missing payload
            {"payload": {}},  # Missing id
            {"id": "2", "payload": {}}  # Missing content in payload
        ]
        
        results = self.search.search("test query")
        self.assertEqual(len(results), 4, "Should return results for all items")
        
        self.assertEqual(results[0].content, "", "Should use empty string for missing content")
        self.assertEqual(results[0].document_id, "", "Should use empty string for missing id")
        self.assertEqual(results[2].document_id, "", "Should use empty string for missing id")
        self.assertEqual(results[3].content, "", "Should use empty string for missing content")

    def test_reranking_with_empty_results(self):
        """Test reranking with empty results."""
        self.mock_vector_store.search.return_value = []
        
        results = self.search.search("test query", rerank=True)
        self.assertEqual(len(results), 0, "Should return empty results when vector store returns none")

    def test_reranking_error_handling(self):
        """Test error handling during reranking."""
        self.mock_vector_store.search.return_value = [
            {"id": "1", "score": 0.9, "payload": {"content": "Test content"}}
        ]
        
        with mock.patch.object(SemanticSearch, "_rerank_results") as mock_rerank:
            mock_rerank.side_effect = Exception("Reranking error")
            
            results = self.search.search("test query", rerank=True)
            self.assertEqual(len(results), 1, "Should return original results on reranking error")
