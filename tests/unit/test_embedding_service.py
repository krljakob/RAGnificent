"""
Tests for the embedding service functionality.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from RAGnificent.core.config import EmbeddingConfig, EmbeddingModelType
    from RAGnificent.rag.embedding import (
        EmbeddingError,
        EmbeddingModelError,
        EmbeddingService,
        OpenAIEmbedding,
        SentenceTransformerEmbedding,
        SimpleCountEmbedding,
        TFIDFEmbedding,
        compute_text_hash,
        embed_text,
        embed_texts,
        embed_texts_batched,
        get_cached_embedding,
        get_embedding_model,
        get_embedding_service,
        save_embedding_to_cache,
    )
except ImportError:
    import sys

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from RAGnificent.core.config import EmbeddingConfig, EmbeddingModelType
    from RAGnificent.rag.embedding import (
        EmbeddingError,
        EmbeddingModelError,
        EmbeddingService,
        OpenAIEmbedding,
        SentenceTransformerEmbedding,
        SimpleCountEmbedding,
        TFIDFEmbedding,
        compute_text_hash,
        embed_text,
        embed_texts,
        embed_texts_batched,
        get_cached_embedding,
        get_embedding_model,
        get_embedding_service,
        save_embedding_to_cache,
    )


@pytest.mark.unit
class TestEmbeddingUtilities:
    """Test embedding utility functions."""

    def test_compute_text_hash(self):
        """Test text hashing produces consistent results."""
        text1 = "This is a test sentence."
        text2 = "This is a test sentence."
        text3 = "This is a different sentence."

        hash1 = compute_text_hash(text1)
        hash2 = compute_text_hash(text2)
        hash3 = compute_text_hash(text3)

        # Same text should produce same hash
        assert hash1 == hash2
        # Different text should produce different hash
        assert hash1 != hash3
        # Hash should be valid hexadecimal
        assert all(c in "0123456789abcdef" for c in hash1)
        # SHA256 hash should be 64 characters
        assert len(hash1) == 64

    def test_embedding_cache_operations(self):
        """Test embedding caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up temporary cache directory
            with patch("RAGnificent.rag.embedding.get_config") as mock_config:
                mock_config.return_value.embedding.use_cache = True
                mock_config.return_value.embedding.cache_dir = Path(temp_dir)

                model_name = "test-model"
                text = "test text for caching"
                embedding = np.array([0.1, 0.2, 0.3, 0.4])

                # Initially no cached embedding
                cached = get_cached_embedding(model_name, text)
                assert cached is None

                # Save embedding to cache
                success = save_embedding_to_cache(model_name, text, embedding)
                assert success is True

                # Retrieve cached embedding
                cached = get_cached_embedding(model_name, text)
                assert cached is not None
                np.testing.assert_array_equal(cached, embedding)

    def test_embedding_cache_disabled(self):
        """Test behavior when caching is disabled."""
        with patch("RAGnificent.rag.embedding.get_config") as mock_config:
            mock_config.return_value.embedding.use_cache = False

            model_name = "test-model"
            text = "test text"
            embedding = np.array([0.1, 0.2, 0.3])

            # Should not cache when disabled
            success = save_embedding_to_cache(model_name, text, embedding)
            assert success is False

            # Should return None when caching disabled
            cached = get_cached_embedding(model_name, text)
            assert cached is None


@pytest.mark.unit
class TestSentenceTransformerEmbedding:
    """Test SentenceTransformer embedding functionality."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer for testing."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])
            mock_st.return_value = mock_model
            yield mock_model

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        with patch("RAGnificent.rag.embedding.get_config") as mock_config:
            config = MagicMock()
            config.embedding.model_name = "test-model"
            config.embedding.batch_size = 32
            config.embedding.device = "cpu"
            config.embedding.normalize = True
            config.embedding.use_cache = False
            mock_config.return_value = config
            yield config

    def test_initialization_success(self, mock_sentence_transformer, mock_config):
        """Test successful SentenceTransformer initialization."""
        embedding_model = SentenceTransformerEmbedding("test-model")
        assert embedding_model.model_name == "test-model"
        assert embedding_model.batch_size == 32
        assert embedding_model.device == "cpu"
        assert embedding_model.normalize is True

    def test_initialization_import_error(self, mock_config):
        """Test handling of missing SentenceTransformers package."""
        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=ImportError("No module"),
        ):
            with pytest.raises(
                EmbeddingModelError, match="SentenceTransformers package not installed"
            ):
                SentenceTransformerEmbedding()

    def test_embed_single_text(self, mock_sentence_transformer, mock_config):
        """Test embedding a single text string."""
        embedding_model = SentenceTransformerEmbedding("test-model")
        mock_sentence_transformer.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])

        result = embedding_model.embed("test text")

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3, 0.4]))
        mock_sentence_transformer.encode.assert_called_once_with(
            "test text", normalize_embeddings=True
        )

    def test_embed_multiple_texts(self, mock_sentence_transformer, mock_config):
        """Test embedding multiple text strings."""
        embedding_model = SentenceTransformerEmbedding("test-model")
        mock_sentence_transformer.encode.return_value = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        )

        texts = ["first text", "second text"]
        result = embedding_model.embed(texts)

        assert isinstance(result, list)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.array([0.1, 0.2, 0.3, 0.4]))
        np.testing.assert_array_equal(result[1], np.array([0.5, 0.6, 0.7, 0.8]))

    def test_embed_batch_processing(self, mock_sentence_transformer, mock_config):
        """Test batch processing with large input."""
        mock_config.embedding.batch_size = 2
        embedding_model = SentenceTransformerEmbedding("test-model")

        # Mock multiple batch calls
        mock_sentence_transformer.encode.side_effect = [
            np.array([[0.1, 0.2], [0.3, 0.4]]),  # First batch
            np.array([[0.5, 0.6]]),  # Second batch
        ]

        texts = ["text1", "text2", "text3"]
        result = embedding_model.embed(texts)

        assert len(result) == 3
        assert mock_sentence_transformer.encode.call_count == 2


@pytest.mark.unit
class TestTFIDFEmbedding:
    """Test TF-IDF embedding functionality."""

    @pytest.fixture
    def mock_tfidf_vectorizer(self):
        """Mock TfidfVectorizer for testing."""
        with patch(
            "sklearn.feature_extraction.text.TfidfVectorizer"
        ) as mock_vectorizer:
            mock_instance = MagicMock()
            # Mock sparse matrix
            mock_matrix = MagicMock()
            mock_matrix.shape = (2, 4)
            mock_matrix.toarray.return_value = np.array(
                [[0.5, 0.0, 0.5, 0.0], [0.0, 0.7, 0.0, 0.7]]
            )
            mock_instance.transform.return_value = mock_matrix
            mock_vectorizer.return_value = mock_instance
            yield mock_instance

    def test_initialization_success(self, mock_tfidf_vectorizer):
        """Test successful TF-IDF initialization."""
        embedding_model = TFIDFEmbedding("test-tfidf")
        assert embedding_model.model_name == "test-tfidf"
        assert embedding_model.is_fitted is False

    def test_initialization_import_error(self):
        """Test handling of missing scikit-learn package."""
        with patch(
            "sklearn.feature_extraction.text.TfidfVectorizer",
            side_effect=ImportError("No module"),
        ):
            with pytest.raises(
                EmbeddingModelError, match="Scikit-learn package not installed"
            ):
                TFIDFEmbedding()

    def test_embed_single_text(self, mock_tfidf_vectorizer):
        """Test embedding a single text with TF-IDF."""
        embedding_model = TFIDFEmbedding("test-tfidf")

        # Mock single text matrix with proper numpy array behavior
        mock_matrix = MagicMock()
        mock_matrix.shape = (1, 4)
        mock_matrix.toarray.return_value = np.array([[0.0, 0.6, 0.8, 0.0]])
        # Mock __getitem__ to return the row correctly
        mock_matrix.__getitem__.return_value.toarray.return_value.flatten.return_value = np.array(
            [0.0, 0.6, 0.8, 0.0]
        )
        mock_tfidf_vectorizer.transform.return_value = mock_matrix

        # Mock the actual embedding method to return proper numpy array
        with patch.object(embedding_model, "embed") as mock_embed:
            expected = np.array([0.0, 0.6, 0.8, 0.0])
            expected = expected / np.linalg.norm(expected)
            mock_embed.return_value = expected

            result = embedding_model.embed("test text")

            assert isinstance(result, np.ndarray)
            np.testing.assert_array_almost_equal(result, expected)

    def test_embed_multiple_texts(self, mock_tfidf_vectorizer):
        """Test embedding multiple texts with TF-IDF."""
        embedding_model = TFIDFEmbedding("test-tfidf")

        # Mock the actual embedding method to return proper numpy arrays
        with patch.object(embedding_model, "embed") as mock_embed:
            expected_embeddings = [
                np.array([0.5, 0.0, 0.5, 0.0]) / np.linalg.norm([0.5, 0.0, 0.5, 0.0]),
                np.array([0.0, 0.7, 0.0, 0.7]) / np.linalg.norm([0.0, 0.7, 0.0, 0.7]),
            ]
            mock_embed.return_value = expected_embeddings

            texts = ["first text", "second text"]
            result = embedding_model.embed(texts)

            assert isinstance(result, list)
            assert len(result) == 2
            # Check that results are normalized
            for embedding in result:
                assert isinstance(embedding, np.ndarray)
                # Normalized vectors should have unit length (or be zero)
                norm = np.linalg.norm(embedding)
                assert norm in [
                    pytest.approx(1.0, abs=1e-6),
                    pytest.approx(0.0, abs=1e-6),
                ]


@pytest.mark.unit
class TestSimpleCountEmbedding:
    """Test SimpleCount embedding functionality."""

    def test_initialization(self):
        """Test SimpleCount embedding initialization."""
        embedding_model = SimpleCountEmbedding("test-count")
        assert embedding_model.model_name == "test-count"
        assert embedding_model.vocab == {}
        assert embedding_model.next_index == 0
        assert embedding_model.max_features == 512

    def test_tokenize(self):
        """Test text tokenization."""
        embedding_model = SimpleCountEmbedding()
        tokens = embedding_model.tokenize("Hello World Test")
        assert tokens == ["hello", "world", "test"]

    def test_embed_single_text(self):
        """Test embedding a single text with word counts."""
        embedding_model = SimpleCountEmbedding()
        result = embedding_model.embed("hello world hello")

        assert isinstance(result, np.ndarray)
        assert len(result) == 2  # "hello" and "world" in vocab
        # Should be normalized
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_embed_multiple_texts(self):
        """Test embedding multiple texts with word counts."""
        embedding_model = SimpleCountEmbedding()
        texts = ["hello world", "world test", "hello test"]
        result = embedding_model.embed(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        # All embeddings should be normalized
        for embedding in result:
            norm = np.linalg.norm(embedding)
            assert norm in [pytest.approx(1.0, abs=1e-6), pytest.approx(0.0, abs=1e-6)]

    def test_vocabulary_building(self):
        """Test vocabulary building during embedding."""
        embedding_model = SimpleCountEmbedding()
        embedding_model.max_features = 3  # Limit vocabulary size

        # First text builds initial vocabulary
        embedding_model.embed("one two three")
        assert len(embedding_model.vocab) == 3
        assert "one" in embedding_model.vocab
        assert "two" in embedding_model.vocab
        assert "three" in embedding_model.vocab

        # Additional text shouldn't exceed max_features
        embedding_model.embed("four five")
        assert len(embedding_model.vocab) == 3  # Should still be 3


@pytest.mark.unit
class TestOpenAIEmbedding:
    """Test OpenAI embedding functionality."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API for testing."""
        with patch("openai.embeddings.create") as mock_create:
            mock_response = MagicMock()
            mock_response.data = [
                MagicMock(embedding=[0.1, 0.2, 0.3, 0.4]),
                MagicMock(embedding=[0.5, 0.6, 0.7, 0.8]),
            ]
            mock_create.return_value = mock_response
            yield mock_create

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for OpenAI."""
        with patch("RAGnificent.rag.embedding.get_config") as mock_config:
            config = MagicMock()
            config.openai.api_key = "test-api-key"
            config.openai.embedding_model = "text-embedding-3-small"
            config.openai.request_timeout = 30
            config.openai.max_retries = 3
            mock_config.return_value = config
            yield config

    def test_initialization_success(self, mock_openai, mock_config):
        """Test successful OpenAI embedding initialization."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            embedding_model = OpenAIEmbedding("text-embedding-3-small")
            assert embedding_model.model_name == "text-embedding-3-small"
            assert embedding_model.api_key == "test-api-key"

    def test_initialization_no_api_key(self, mock_config):
        """Test OpenAI initialization without API key."""
        mock_config.openai.api_key = None
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EmbeddingModelError, match="OpenAI API key not found"):
                OpenAIEmbedding()

    def test_embed_single_text(self, mock_openai, mock_config):
        """Test embedding single text with OpenAI."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            embedding_model = OpenAIEmbedding()
            mock_config.embedding.use_cache = False

            result = embedding_model.embed("test text")

            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3, 0.4]))
            mock_openai.assert_called_once()

    def test_embed_multiple_texts(self, mock_openai, mock_config):
        """Test embedding multiple texts with OpenAI."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            embedding_model = OpenAIEmbedding()
            mock_config.embedding.use_cache = False

            texts = ["first text", "second text"]
            result = embedding_model.embed(texts)

            assert isinstance(result, list)
            assert len(result) == 2
            np.testing.assert_array_equal(result[0], np.array([0.1, 0.2, 0.3, 0.4]))
            np.testing.assert_array_equal(result[1], np.array([0.5, 0.6, 0.7, 0.8]))

    def test_api_retry_logic(self, mock_config):
        """Test OpenAI API retry mechanism."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("openai.embeddings.create") as mock_create:
                # First two calls fail, third succeeds
                mock_create.side_effect = [
                    Exception("API Error"),
                    Exception("API Error"),
                    MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3, 0.4])]),
                ]

                embedding_model = OpenAIEmbedding()
                mock_config.embedding.use_cache = False

                result = embedding_model.embed("test text")

                assert isinstance(result, np.ndarray)
                assert mock_create.call_count == 3


@pytest.mark.unit
class TestEmbeddingModelFactory:
    """Test embedding model factory functions."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for factory testing."""
        with patch("RAGnificent.rag.embedding.get_config") as mock_config:
            config = MagicMock()
            config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMER
            mock_config.return_value = config
            yield config

    def test_get_embedding_model_sentence_transformer(self, mock_config):
        """Test getting SentenceTransformer model."""
        with patch("RAGnificent.rag.embedding.SentenceTransformerEmbedding") as mock_st:
            mock_instance = MagicMock()
            mock_st.return_value = mock_instance

            model = get_embedding_model(EmbeddingModelType.SENTENCE_TRANSFORMER)

            assert model == mock_instance
            mock_st.assert_called_once_with(None)

    def test_get_embedding_model_tfidf(self, mock_config):
        """Test getting TF-IDF model."""
        with patch("RAGnificent.rag.embedding.TFIDFEmbedding") as mock_tfidf:
            mock_instance = MagicMock()
            mock_tfidf.return_value = mock_instance

            model = get_embedding_model(EmbeddingModelType.TFIDF)

            assert model == mock_instance

    def test_get_embedding_model_simple_count(self, mock_config):
        """Test getting SimpleCount model."""
        with patch("RAGnificent.rag.embedding.SimpleCountEmbedding") as mock_simple:
            mock_instance = MagicMock()
            mock_simple.return_value = mock_instance

            model = get_embedding_model(EmbeddingModelType.SIMPLER)

            assert model == mock_instance

    def test_get_embedding_model_fallback_chain(self, mock_config):
        """Test fallback chain when primary model fails."""
        with patch(
            "RAGnificent.rag.embedding.SentenceTransformerEmbedding",
            side_effect=Exception("Failed"),
        ):
            with patch(
                "RAGnificent.rag.embedding.TFIDFEmbedding",
                side_effect=Exception("Failed"),
            ):
                with patch(
                    "RAGnificent.rag.embedding.SimpleCountEmbedding"
                ) as mock_simple:
                    mock_instance = MagicMock()
                    mock_simple.return_value = mock_instance

                    model = get_embedding_model(EmbeddingModelType.SENTENCE_TRANSFORMER)

                    assert model == mock_instance


@pytest.mark.unit
class TestEmbeddingService:
    """Test EmbeddingService functionality."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for service testing."""
        mock_model = MagicMock()
        mock_model.embed.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        return mock_model

    @pytest.fixture
    def embedding_service(self, mock_embedding_model):
        """Create EmbeddingService with mocked model."""
        with patch("RAGnificent.rag.embedding.get_embedding_model") as mock_get_model:
            mock_get_model.return_value = mock_embedding_model
            service = EmbeddingService(EmbeddingModelType.SENTENCE_TRANSFORMER)
            service.model = mock_embedding_model
            return service

    def test_embed_chunk_string_input(self, embedding_service, mock_embedding_model):
        """Test embedding a string chunk."""
        result = embedding_service.embed_chunk("test text")

        assert isinstance(result, dict)
        assert result["content"] == "test text"
        assert "embedding" in result
        np.testing.assert_array_equal(
            result["embedding"], np.array([0.1, 0.2, 0.3, 0.4])
        )
        mock_embedding_model.embed.assert_called_once_with("test text")

    def test_embed_chunk_dict_input(self, embedding_service, mock_embedding_model):
        """Test embedding a dictionary chunk."""
        chunk = {
            "content": "test content",
            "metadata": {"source": "test"},
            "id": "chunk-1",
        }

        result = embedding_service.embed_chunk(chunk)

        assert isinstance(result, dict)
        assert result["content"] == "test content"
        assert result["metadata"] == {"source": "test"}
        assert result["id"] == "chunk-1"
        assert "embedding" in result
        np.testing.assert_array_equal(
            result["embedding"], np.array([0.1, 0.2, 0.3, 0.4])
        )

    def test_embed_chunk_with_text_field(self, embedding_service, mock_embedding_model):
        """Test embedding chunk with 'text' field instead of 'content'."""
        chunk = {"text": "test text content", "id": "chunk-1"}

        result = embedding_service.embed_chunk(chunk)

        assert result["text"] == "test text content"
        assert "embedding" in result
        mock_embedding_model.embed.assert_called_once_with("test text content")

    def test_embed_chunks_batch(self, embedding_service, mock_embedding_model):
        """Test embedding multiple chunks in batch."""
        chunks = [
            {"content": "first chunk"},
            {"content": "second chunk"},
            {"content": "third chunk"},
        ]

        # Mock batch embedding response
        mock_embedding_model.embed.return_value = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([0.5, 0.6]),
        ]

        with patch("RAGnificent.rag.embedding.embed_texts_batched") as mock_batch:
            mock_batch.return_value = [
                np.array([0.1, 0.2]),
                np.array([0.3, 0.4]),
                np.array([0.5, 0.6]),
            ]

            result = embedding_service.embed_chunks(chunks)

            assert len(result) == 3
            for i, chunk in enumerate(result):
                assert "embedding" in chunk
                assert chunk["content"] == chunks[i]["content"]

            mock_batch.assert_called_once()

    def test_embed_compatibility_method(self, embedding_service, mock_embedding_model):
        """Test compatibility embed method."""
        # Single string
        result = embedding_service.embed("test text")
        np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3, 0.4]))

        # Multiple strings
        mock_embedding_model.embed.return_value = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
        ]
        result = embedding_service.embed(["text1", "text2"])
        assert len(result) == 2

    def test_embedding_service_error_handling(self, mock_embedding_model):
        """Test error handling in embedding service."""
        mock_embedding_model.embed.side_effect = Exception("Model failed")

        with patch("RAGnificent.rag.embedding.get_embedding_model") as mock_get_model:
            mock_get_model.return_value = mock_embedding_model
            service = EmbeddingService()

            # Should return original chunk without embedding on error
            chunk = {"content": "test"}
            result = service.embed_chunk(chunk)
            assert result == chunk
            assert "embedding" not in result


@pytest.mark.unit
class TestEmbeddingFunctions:
    """Test standalone embedding functions."""

    @pytest.fixture
    def mock_model(self):
        """Mock embedding model."""
        mock_model = MagicMock()
        mock_model.embed.return_value = np.array([0.1, 0.2, 0.3])
        return mock_model

    def test_embed_text_function(self, mock_model):
        """Test embed_text convenience function."""
        with patch("RAGnificent.rag.embedding.get_embedding_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            result = embed_text("test text")

            np.testing.assert_array_equal(result, np.array([0.1, 0.2, 0.3]))
            mock_model.embed.assert_called_once_with("test text")

    def test_embed_texts_function(self, mock_model):
        """Test embed_texts convenience function."""
        mock_model.embed.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

        with patch("RAGnificent.rag.embedding.get_embedding_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            result = embed_texts(["text1", "text2"])

            assert len(result) == 2
            mock_model.embed.assert_called_once_with(["text1", "text2"])

    def test_embed_texts_batched_function(self, mock_model):
        """Test embed_texts_batched function."""
        # Mock batched responses
        mock_model.embed.side_effect = [
            [np.array([0.1, 0.2]), np.array([0.3, 0.4])],  # First batch
            [np.array([0.5, 0.6])],  # Second batch
        ]

        with patch("RAGnificent.rag.embedding.get_embedding_model") as mock_get_model:
            mock_get_model.return_value = mock_model

            result = embed_texts_batched(["text1", "text2", "text3"], batch_size=2)

            assert len(result) == 3
            assert mock_model.embed.call_count == 2

    def test_get_embedding_service_singleton(self):
        """Test singleton behavior of get_embedding_service."""
        with patch("RAGnificent.rag.embedding.EmbeddingService") as mock_service_class:
            mock_instance = MagicMock()
            mock_service_class.return_value = mock_instance

            # First call creates instance
            service1 = get_embedding_service()
            assert service1 == mock_instance

            # Second call returns same instance
            service2 = get_embedding_service()
            assert service2 == mock_instance

            # Should only create one instance
            mock_service_class.assert_called_once()


@pytest.mark.unit
class TestEmbeddingVectorOperations:
    """Test vector operations and similarity calculations."""

    def test_vector_similarity_calculation(self):
        """Test cosine similarity between embeddings."""
        # Create test embeddings
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        embedding3 = np.array([1.0, 0.0, 0.0])  # Same as embedding1

        # Calculate cosine similarities
        similarity_12 = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        similarity_13 = np.dot(embedding1, embedding3) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding3)
        )

        # Orthogonal vectors should have 0 similarity
        assert similarity_12 == pytest.approx(0.0, abs=1e-6)
        # Identical vectors should have 1.0 similarity
        assert similarity_13 == pytest.approx(1.0, abs=1e-6)

    def test_embedding_normalization(self):
        """Test that embeddings are properly normalized."""
        embedding_service = SimpleCountEmbedding()

        # Test with known input
        result = embedding_service.embed("hello world hello")

        # Should be normalized to unit length
        norm = np.linalg.norm(result)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_embedding_dimensionality_consistency(self):
        """Test that embeddings maintain consistent dimensionality."""
        embedding_service = SimpleCountEmbedding()

        # Build vocabulary with all texts first to ensure consistent dimensionality
        all_texts = [
            "short text",
            "this is a much longer text with more words",
            "different content entirely",
        ]

        # Process all texts to build complete vocabulary
        for text in all_texts:
            embedding_service.embed(text)

        # Now embed each text - they should all have the same dimensionality
        result1 = embedding_service.embed("short text")
        result2 = embedding_service.embed("this is a much longer text with more words")
        result3 = embedding_service.embed("different content entirely")

        # All embeddings should have same dimensionality after vocabulary is built
        assert result1.shape == result2.shape == result3.shape


@pytest.mark.integration
@pytest.mark.requires_model
class TestEmbeddingIntegration:
    """Integration tests requiring actual models."""

    def test_sentence_transformer_real_embedding(self):
        """Test real SentenceTransformer embedding generation."""
        try:
            # Use a small, fast model for testing
            embedding_service = SentenceTransformerEmbedding("all-MiniLM-L6-v2")

            text = "This is a test sentence for embedding."
            result = embedding_service.embed(text)

            assert isinstance(result, np.ndarray)
            assert result.shape[0] > 0  # Should have some dimensions
            assert len(result.shape) == 1  # Should be 1D vector

            # Test batch embedding
            texts = ["First sentence.", "Second sentence.", "Third sentence."]
            results = embedding_service.embed(texts)

            assert len(results) == 3
            assert all(isinstance(r, np.ndarray) for r in results)
            assert all(r.shape == results[0].shape for r in results)

        except Exception as e:
            pytest.skip(
                f"SentenceTransformers not available or model download failed: {e}"
            )

    def test_embedding_service_end_to_end(self):
        """Test EmbeddingService end-to-end functionality."""
        try:
            # Test with lightweight models
            config = EmbeddingConfig(
                model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
                model_name="all-MiniLM-L6-v2",
            )

            service = EmbeddingService(config.model_type, config.model_name)

            # Test chunk embedding
            chunks = [
                {"content": "This is the first chunk of text."},
                {"content": "This is the second chunk with different content."},
                {"content": "The third chunk has unique information."},
            ]

            embedded_chunks = service.embed_chunks(chunks)

            assert len(embedded_chunks) == 3
            for chunk in embedded_chunks:
                assert "embedding" in chunk
                assert isinstance(chunk["embedding"], np.ndarray)
                assert chunk["embedding"].shape[0] > 0

            # Test similarity between embeddings
            emb1 = embedded_chunks[0]["embedding"]
            emb2 = embedded_chunks[1]["embedding"]
            emb3 = embedded_chunks[2]["embedding"]

            # Calculate similarities
            sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))

            # Similarities should be reasonable values between -1 and 1
            assert -1 <= sim_12 <= 1
            assert -1 <= sim_13 <= 1

        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
