"""
Test edge cases in the embedding module.
"""

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

project_root = Path(__file__).parent.parent.parent
rag_path = project_root / "RAGnificent" / "rag"
sys.path.insert(0, str(project_root))

from RAGnificent.rag.embedding import (
    EmbeddingError,
    EmbeddingModelError,
    EmbeddingAPIError,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    TFIDFEmbedding,
    SimpleCountEmbedding,
    get_embedding_model,
    get_embedding_service,
    compute_text_hash,
    get_cached_embedding,
    save_embedding_to_cache
)
from RAGnificent.core.config import EmbeddingModelType


class TestEmbeddingEdgeCases(unittest.TestCase):
    """Test edge cases for the embedding module."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name) / "cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache_patcher = mock.patch(
            "RAGnificent.rag.embedding.get_config"
        )
        self.mock_config = self.cache_patcher.start()
        self.mock_config.return_value.embedding.cache_dir = self.cache_dir
        self.mock_config.return_value.embedding.use_cache = True
        self.mock_config.return_value.embedding.model_name = "test-model"
        self.mock_config.return_value.embedding.batch_size = 8
        self.mock_config.return_value.embedding.device = "cpu"
        self.mock_config.return_value.embedding.normalize = True
        
        self.test_text = "This is a test text for embedding."
        self.test_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    def tearDown(self):
        """Clean up test environment."""
        self.cache_patcher.stop()
        self.temp_dir.cleanup()

    def test_corrupted_cache(self):
        """Test handling of corrupted cache files."""
        text_hash = compute_text_hash(self.test_text)
        model_name = "test-model"
        model_cache_dir = self.cache_dir / model_name.replace("/", "_")
        os.makedirs(model_cache_dir, exist_ok=True)
        
        cache_path = model_cache_dir / f"{text_hash}.pkl"
        with open(cache_path, "wb") as f:
            f.write(b"corrupted data")
        
        result = get_cached_embedding(model_name, self.test_text)
        self.assertIsNone(result, "Corrupted cache should return None")

    def test_cache_directory_creation(self):
        """Test cache directory creation for new models."""
        model_name = "org/model/with/slashes"
        text_hash = compute_text_hash(self.test_text)
        
        save_embedding_to_cache(model_name, self.test_text, self.test_embedding)
        
        model_cache_dir = self.cache_dir / model_name.replace("/", "_")
        self.assertTrue(model_cache_dir.exists(), "Cache directory should be created")
        
        cache_path = model_cache_dir / f"{text_hash}.pkl"
        self.assertTrue(cache_path.exists(), "Cache file should be created")
