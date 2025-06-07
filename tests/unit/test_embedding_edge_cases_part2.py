"""
Test edge cases in the embedding module (part 2).
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

try:
    from RAGnificent.rag.embedding import (
        EmbeddingAPIError,
        EmbeddingModelError,
        OpenAIEmbedding,
        SentenceTransformerEmbedding,
    )
except ImportError:
    project_root = Path(__file__).parent.parent.parent
    rag_path = project_root / "RAGnificent" / "rag"
    sys.path.insert(0, str(project_root))
    from RAGnificent.rag.embedding import (
        EmbeddingAPIError,
        EmbeddingModelError,
        OpenAIEmbedding,
        SentenceTransformerEmbedding,
    )


class TestEmbeddingEdgeCasesPart2(unittest.TestCase):
    """Test edge cases for the embedding module (part 2)."""

    def setUp(self):
        """Set up test environment."""
        self.config_patcher = mock.patch("RAGnificent.rag.embedding.get_config")
        self.mock_config = self.config_patcher.start()
        self.mock_config.return_value.embedding.model_name = "test-model"
        self.mock_config.return_value.embedding.batch_size = 8
        self.mock_config.return_value.embedding.device = "cpu"
        self.mock_config.return_value.embedding.normalize = True
        self.mock_config.return_value.openai.api_key = "test_key"
        self.mock_config.return_value.openai.embedding_model = "text-embedding-3-small"
        self.mock_config.return_value.openai.request_timeout = 10
        self.mock_config.return_value.openai.max_retries = 2

    def tearDown(self):
        """Clean up test environment."""
        self.config_patcher.stop()

    @mock.patch("sentence_transformers.SentenceTransformer", create=True)
    def test_sentence_transformer_import_error(self, mock_st):
        """Test handling of SentenceTransformer import error."""
        mock_st.side_effect = ImportError("No module named 'sentence_transformers'")

        with self.assertRaises(EmbeddingModelError):
            SentenceTransformerEmbedding()

    @mock.patch("sentence_transformers.SentenceTransformer", create=True)
    def test_sentence_transformer_model_error(self, mock_st):
        """Test handling of SentenceTransformer model loading error."""
        mock_st.side_effect = Exception("Model not found")

        with self.assertRaises(EmbeddingModelError):
            SentenceTransformerEmbedding()

    @mock.patch("sentence_transformers.SentenceTransformer", create=True)
    def test_sentence_transformer_embedding_error(self, mock_st):
        """Test handling of SentenceTransformer embedding error."""
        mock_model = mock.MagicMock()
        mock_model.encode.side_effect = Exception("Encoding error")
        mock_st.return_value = mock_model

        embedding_model = SentenceTransformerEmbedding()

        with self.assertRaises(EmbeddingModelError):
            embedding_model.embed("Test text")

    def test_openai_missing_api_key(self):
        """Test handling of missing OpenAI API key."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            self.mock_config.return_value.openai.api_key = None

            with self.assertRaises(EmbeddingModelError):
                OpenAIEmbedding()

    @mock.patch("openai.embeddings.create", create=True)
    def test_openai_api_error(self, mock_openai):
        """Test handling of OpenAI API error."""
        mock_openai.side_effect = Exception("API error")

        embedding_model = OpenAIEmbedding()

        with self.assertRaises(EmbeddingAPIError):
            embedding_model.embed("Test text")

    @mock.patch("openai.embeddings.create", create=True)
    def test_openai_retry_logic(self, mock_openai):
        """Test OpenAI retry logic for transient errors."""
        mock_response = mock.MagicMock()
        mock_response.data = [mock.MagicMock(embedding=[0.1, 0.2, 0.3])]

        mock_openai.side_effect = [
            Exception("Transient error"),
            mock_response,
        ]

        embedding_model = OpenAIEmbedding()

        result = embedding_model.embed("Test text")

        self.assertIsNotNone(result, "Should return embedding after successful retry")
        self.assertEqual(mock_openai.call_count, 2, "Should have called API twice")
