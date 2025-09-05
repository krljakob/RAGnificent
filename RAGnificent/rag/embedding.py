"""
Embedding module for RAGnificent.

Provides functions for generating and managing vector embeddings using various models.
Supports multiple embedding strategies including SentenceTransformers, OpenAI, and fallback options.
"""

import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dotenv import load_dotenv

from ..core.config import EmbeddingModelType, get_config

logger = logging.getLogger(__name__)

load_dotenv()


class EmbeddingError(Exception):
    """Base exception for embedding generation"""

    pass


class EmbeddingModelError(EmbeddingError):
    """Exception when embedding model fails"""

    pass


class EmbeddingAPIError(EmbeddingError):
    """Exception for API errors during embedding"""

    pass


def get_embedding_cache_path(model_name: str, text_hash: str) -> Path:
    """Get path for cached embeddings."""
    import re

    config = get_config()
    cache_dir = config.embedding.cache_dir

    safe_model_name = re.sub(r"[^\w\-_.]", "_", model_name)
    safe_model_name = safe_model_name.replace("..", "_")
    safe_model_name = safe_model_name.strip(".") or "default_model"

    model_cache_dir = cache_dir / safe_model_name
    os.makedirs(model_cache_dir, exist_ok=True)

    if not re.match(r"^[a-f0-9]+$", text_hash):
        raise ValueError(f"Invalid text hash format: {text_hash}")

    return model_cache_dir / f"{text_hash}.npy"


def compute_text_hash(text: str) -> str:
    """Compute deterministic hash for text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cached_embedding(model_name: str, text: str) -> Optional[np.ndarray]:
    """
    Get embedding from cache if available

    Args:
        model_name: Name of the embedding model
        text: Text content

    Returns:
        Cached embedding if available, None otherwise
    """
    config = get_config()

    if not config.embedding.use_cache:
        return None

    text_hash = compute_text_hash(text)
    cache_path = get_embedding_cache_path(model_name, text_hash)

    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cached embedding: {e}")

    return None


def save_embedding_to_cache(model_name: str, text: str, embedding: np.ndarray) -> bool:
    """Save embedding to cache."""
    config = get_config()

    if not config.embedding.use_cache:
        return False

    text_hash = compute_text_hash(text)
    cache_path = get_embedding_cache_path(model_name, text_hash)

    try:
        np.save(cache_path, embedding)
        return True
    except Exception as e:
        logger.warning(f"Failed to save embedding to cache: {e}")
        return False


class SentenceTransformerEmbedding:
    """Embedding generation using SentenceTransformers."""

    def __init__(self, model_name: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer

            config = get_config()
            self.model_name = model_name or config.embedding.model_name
            self.batch_size = config.embedding.batch_size
            self.device = config.embedding.device
            self.normalize = config.embedding.normalize

            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Successfully loaded SentenceTransformer model")

        except ImportError as e:
            logger.error(
                "SentenceTransformers package not installed. Please install with: uv pip install sentence-transformers"
            )
            raise EmbeddingModelError(
                "SentenceTransformers package not installed"
            ) from e

        except Exception as e:
            logger.error(f"Error loading SentenceTransformer model: {e}")
            raise EmbeddingModelError(
                f"Failed to load SentenceTransformer model: {e}"
            ) from e

    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings for text input."""
        try:
            if isinstance(text, str):
                cached = get_cached_embedding(self.model_name, text)
                if cached is not None:
                    return cached

                embedding = self.model.encode(text, normalize_embeddings=self.normalize)

                save_embedding_to_cache(self.model_name, text, embedding)
                return embedding

            embeddings = []
            texts_to_embed = []
            text_indices = []

            for i, t in enumerate(text):
                cached = get_cached_embedding(self.model_name, t)
                if cached is not None:
                    embeddings.append(cached)
                else:
                    texts_to_embed.append(t)
                    text_indices.append(i)

            if not texts_to_embed:
                return embeddings

            new_embeddings = []
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch_texts = texts_to_embed[i : i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    normalize_embeddings=self.normalize,
                    batch_size=self.batch_size,
                )
                new_embeddings.extend(batch_embeddings)

            for i, embedding in enumerate(new_embeddings):
                save_embedding_to_cache(self.model_name, texts_to_embed[i], embedding)

            result = [None] * len(text)
            for i, embed in zip(text_indices, new_embeddings, strict=False):
                result[i] = embed

            embedding_iter = iter(embeddings)
            for i in range(len(result)):
                if result[i] is None:
                    try:
                        result[i] = next(embedding_iter)
                    except StopIteration:
                        break

            return result

        except Exception as e:
            logger.error(f"Error generating embeddings with SentenceTransformer: {e}")
            raise EmbeddingModelError(f"Failed to generate embeddings: {e}") from e


class OpenAIEmbedding:
    """Embedding generation using OpenAI API"""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize OpenAI embedding model using the official client.

        Args:
            model_name: Model name to use
        """
        try:
            import httpx
            from openai import OpenAI

            config = get_config()
            self.api_key = config.openai.api_key or os.getenv("OPENAI_API_KEY")
            self.model_name = model_name or config.openai.embedding_model
            self.request_timeout = config.openai.request_timeout
            self.max_retries = config.openai.max_retries

            if not self.api_key:
                raise EmbeddingModelError("OpenAI API key not found")

            timeout = httpx.Timeout(self.request_timeout)
            http_client = httpx.Client(timeout=timeout)
            self.client = OpenAI(api_key=self.api_key, http_client=http_client)
            logger.info(f"Initialized OpenAI embedding with model: {self.model_name}")

        except ImportError as e:
            logger.error(
                "OpenAI package not installed. Please install with: uv pip install openai httpx"
            )
            raise EmbeddingModelError("OpenAI package not installed") from e
        except Exception as e:
            logger.error(f"Error initializing OpenAI embedding: {e}")
            raise EmbeddingModelError(
                f"Failed to initialize OpenAI embedding: {e}"
            ) from e

    def _get_cached_embeddings(
        self, texts: List[str], is_single_text: bool
    ) -> Tuple[List[np.ndarray], List[str], List[int]]:
        """
        Check cache for embeddings and return cached ones plus texts that need embedding.

        Args:
            texts: List of text strings to check in cache
            is_single_text: Whether this originated from a single text string

        Returns:
            Tuple of (cached_embeddings, texts_to_embed, original_indices)
        """
        if is_single_text:
            cached = get_cached_embedding(self.model_name, texts[0])
            return ([cached], [], []) if cached is not None else ([], texts, [])
        # For list input, check cache for each text
        cached_embeddings = []
        texts_to_embed = []
        original_indices = []

        for i, t in enumerate(texts):
            cached = get_cached_embedding(self.model_name, t)
            if cached is not None:
                cached_embeddings.append(cached)
            else:
                texts_to_embed.append(t)
                original_indices.append(i)

        return cached_embeddings, texts_to_embed, original_indices

    def _call_openai_api(self, texts: List[str]) -> List[np.ndarray]:
        """
        Call OpenAI API with retry logic to get embeddings.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        # Retry logic for API calls
        retry_count = 0
        while retry_count <= self.max_retries:
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )
                # Process response
                return [np.array(item.embedding) for item in response.data]
            except Exception as e:
                retry_count += 1
                if retry_count > self.max_retries:
                    raise
                logger.warning(
                    f"Retrying OpenAI embedding request ({retry_count}/{self.max_retries}): {e}"
                )
                time.sleep(2**retry_count)  # Exponential backoff
        return None

    def _cache_embeddings(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        original_text: Union[str, List[str]],
    ) -> None:
        """
        Save embeddings to cache.

        Args:
            texts: The texts that were embedded
            embeddings: The embedding vectors
            original_text: The original input text (single string or list)
        """
        if len(texts) != len(embeddings):
            logger.error(f"Mismatch between texts ({len(texts)}) and embeddings ({len(embeddings)}) lengths")
            return

        for i, t in enumerate(texts):
            if isinstance(original_text, str) or t not in (
                original_text if isinstance(original_text, list) else [original_text]
            ):
                save_embedding_to_cache(self.model_name, t, embeddings[i])

    def _merge_embeddings(
        self,
        new_embeddings: List[np.ndarray],
        cached_embeddings: List[np.ndarray],
        original_indices: List[int],
        total_length: int,
    ) -> List[np.ndarray]:
        """
        Merge cached and new embeddings into the correct order.

        Args:
            new_embeddings: List of newly generated embeddings
            cached_embeddings: List of embeddings retrieved from cache
            original_indices: Original indices of texts that needed embedding
            total_length: Total number of texts in the original input

        Returns:
            List of embeddings in the original order
        """
        result = [None] * total_length

        # Place new embeddings in their original positions
        for original_idx, embed in zip(original_indices, new_embeddings, strict=False):
            result[original_idx] = embed

        # Fill in gaps with cached embeddings
        cached_idx = 0
        for i in range(total_length):
            if result[i] is None:
                result[i] = cached_embeddings[cached_idx]
                cached_idx += 1

        return result

    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text input using OpenAI API.

        Args:
            text: Single text string or list of text strings

        Returns:
            Embedding vector(s)
        """
        try:
            # Prepare input for consistent processing
            is_single_text = isinstance(text, str)
            texts = [text] if is_single_text else text

            # Check cache and prepare texts for embedding
            (
                cached_embeddings,
                texts_to_embed,
                original_indices,
            ) = self._get_cached_embeddings(texts, is_single_text)

            if not texts_to_embed:
                if not is_single_text:
                    return cached_embeddings

                # Everything was in cache
                return cached_embeddings[0] if is_single_text else cached_embeddings

            new_embeddings = self._call_openai_api(texts_to_embed)
            if new_embeddings is None:
                raise EmbeddingAPIError("Failed to get embeddings from OpenAI API after all retries")

            # Cache the new embeddings
            self._cache_embeddings(texts_to_embed, new_embeddings, text)
            # Return results in the appropriate format
            if is_single_text:
                return new_embeddings[0]

            # For list input, merge cached and new embeddings
            if original_indices:
                return self._merge_embeddings(
                    new_embeddings, cached_embeddings, original_indices, len(texts)
                )

            return new_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise EmbeddingAPIError(f"Failed to generate OpenAI embeddings: {e}") from e


class TFIDFEmbedding:
    """Fallback embedding using TF-IDF"""

    def __init__(self, model_name: str = "tfidf"):
        """
        Initialize TF-IDF embedding model.

        Args:
            model_name: Name identifier for this model
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.model_name = model_name
            self.vectorizer = TfidfVectorizer(max_features=1024, stop_words="english")
            self.is_fitted = False
            logger.info("Initialized TF-IDF embedding model")

        except ImportError as e:
            logger.error(
                "Scikit-learn package not installed. Please install with: uv pip install scikit-learn"
            )
            raise EmbeddingModelError("Scikit-learn package not installed") from e

        except Exception as e:
            logger.error(f"Error initializing TF-IDF embedding: {e}")
            raise EmbeddingModelError(
                f"Failed to initialize TF-IDF embedding: {e}"
            ) from e

    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text input using TF-IDF.

        Args:
            text: Single text string or list of text strings

        Returns:
            Embedding vector(s)
        """
        try:
            texts = [text] if isinstance(text, str) else text

            # Fit vectorizer if not already fitted
            if not self.is_fitted:
                self.vectorizer.fit(texts)
                self.is_fitted = True

            # Transform texts to vectors
            tfidf_matrix = self.vectorizer.transform(texts)

            # Convert to normalized dense arrays
            embeddings = []
            for i in range(tfidf_matrix.shape[0]):
                vec = tfidf_matrix[i].toarray().flatten()
                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embeddings.append(vec)

            # Return single embedding or list
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with TF-IDF: {e}")
            raise EmbeddingModelError(
                f"Failed to generate TF-IDF embeddings: {e}"
            ) from e


class SimpleCountEmbedding:
    """Ultra-simple fallback embedding using word counts"""

    def __init__(self, model_name: str = "simple_count"):
        """
        Initialize simple count embedding model.

        Args:
            model_name: Name identifier for this model
        """
        self.model_name = model_name
        self.vocab = {}
        self.next_index = 0
        self.max_features = 512
        logger.info("Initialized simple count embedding model")

    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        return text.lower().split()

    def embed(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text input using simple word counts.

        Args:
            text: Single text string or list of text strings

        Returns:
            Embedding vector(s)
        """
        try:
            texts = [text] if isinstance(text, str) else text

            # Build vocabulary if needed
            for t in texts:
                tokens = self.tokenize(t)
                for token in tokens:
                    if token not in self.vocab and len(self.vocab) < self.max_features:
                        self.vocab[token] = self.next_index
                        self.next_index += 1

            # Generate count vectors
            embeddings = []
            for t in texts:
                vec = np.zeros(len(self.vocab))
                tokens = self.tokenize(t)
                for token in tokens:
                    if token in self.vocab:
                        vec[self.vocab[token]] += 1

                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embeddings.append(vec)

            # Return single embedding or list
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with simple count: {e}")
            raise EmbeddingModelError(
                f"Failed to generate simple count embeddings: {e}"
            ) from e


def get_embedding_model(
    model_type: Optional[EmbeddingModelType] = None, model_name: Optional[str] = None
):
    """
    Get appropriate embedding model based on configuration.

    Args:
        model_type: Type of embedding model to use
        model_name: Specific model name to use

    Returns:
        Embedding model instance
    """
    config = get_config()
    model_type = model_type or config.embedding.model_type

    try:
        if model_type == EmbeddingModelType.SENTENCE_TRANSFORMER:
            return SentenceTransformerEmbedding(model_name)

        if model_type == EmbeddingModelType.OPENAI:
            return OpenAIEmbedding(model_name)

        if model_type == EmbeddingModelType.TFIDF:
            return TFIDFEmbedding()

        if model_type == EmbeddingModelType.SIMPLER:
            return SimpleCountEmbedding()

        # Default to SentenceTransformer
        logger.warning(
            f"Unknown model type '{model_type}', defaulting to SentenceTransformer"
        )
        return SentenceTransformerEmbedding(model_name)

    except Exception as e:
        logger.error(f"Failed to initialize {model_type} embedding model: {e}")

        # Fallback chain: ST -> TF-IDF -> SimpleCount
        try:
            logger.warning("Falling back to TF-IDF embedding model")
            return TFIDFEmbedding()
        except Exception:
            logger.warning("Falling back to simple count embedding model")
            return SimpleCountEmbedding()


def embed_text(text: str, model=None) -> np.ndarray:
    """
    Generate embedding for a single text string.

    Args:
        text: Text to embed
        model: Optional pre-initialized embedding model

    Returns:
        Embedding vector
    """
    if model is None:
        model = get_embedding_model()

    return model.embed(text)


def embed_texts(texts: List[str], model=None) -> List[np.ndarray]:
    """
    Generate embeddings for multiple text strings.

    Args:
        texts: List of texts to embed
        model: Optional pre-initialized embedding model

    Returns:
        List of embedding vectors
    """
    if model is None:
        model = get_embedding_model()

    return model.embed(texts)


def embed_texts_batched(
    texts: List[str], model=None, batch_size: int = None
) -> List[np.ndarray]:
    """
    Generate embeddings for multiple text strings in batches for efficiency.

    Args:
        texts: List of texts to embed
        model: Optional pre-initialized embedding model
        batch_size: Size of batches to process (uses config if None)

    Returns:
        List of embedding vectors
    """
    if model is None:
        model = get_embedding_model()

    if batch_size is None:
        config = get_config()
        batch_size = config.embedding.batch_size

    # Process in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.embed(batch)
        all_embeddings.extend(embeddings)

    return all_embeddings


class EmbeddingService:
    """Service for generating embeddings for text chunks."""

    def __init__(
        self,
        model_type: Optional[EmbeddingModelType] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the embedding service.

        Args:
            model_type: Type of embedding model to use
            model_name: The name of the embedding model to use
        """
        try:
            self.model = get_embedding_model(model_type, model_name)
            logger.info(
                f"Initialized embedding service with model: {type(self.model).__name__}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def embed_chunk(self, chunk: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an embedding for a single chunk.

        Args:
            chunk: Either a string text chunk or a Chunk object dictionary

        Returns:
            The chunk with an added embedding field
        """
        if isinstance(chunk, str):
            text = chunk
            return {"content": text, "embedding": self.model.embed(text)}

        # If it's a chunk object
        try:
            content = chunk.get("content", "")
            if not content and "text" in chunk:
                content = chunk["text"]

            # Generate embedding
            embedding = self.model.embed(content)

            # Add embedding to the chunk
            return {**chunk, "embedding": embedding}

        except Exception as e:
            logger.error(f"Error embedding chunk: {e}")
            # Return original chunk without embedding in case of error
            return chunk

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple chunks with batching for efficiency.

        Args:
            chunks: List of chunk objects

        Returns:
            List of chunks with embeddings added
        """
        try:
            # Extract text content for batched embedding
            texts = []
            for chunk in chunks:
                content = chunk.get("content", "")
                if not content and "text" in chunk:
                    content = chunk["text"]
                texts.append(content)

            # Generate embeddings in batch
            embeddings = embed_texts_batched(texts, self.model)

            # Add embeddings back to chunks
            result = []
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    result.append({**chunk, "embedding": embeddings[i]})
                else:
                    # This should not happen if batching works correctly
                    logger.warning(f"Missing embedding for chunk {i}")
                    result.append(chunk)

            return result

        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            return chunks  # Return original chunks without embeddings in case of error

    def embed(
        self, texts: Union[str, List[str]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s) - compatibility method for tests.

        Args:
            texts: String or list of strings to embed

        Returns:
            Single embedding array or list of embedding arrays
        """
        return self.model.embed(texts)


# Singleton instance for easy access
_default_embedding_service = None


def get_embedding_service(
    model_type: Optional[EmbeddingModelType] = None, model_name: Optional[str] = None
) -> EmbeddingService:
    """
    Get or create the default embedding service.

    Args:
        model_type: Type of embedding model to use
        model_name: The name of the embedding model to use

    Returns:
        The embedding service instance
    """
    global _default_embedding_service
    if _default_embedding_service is None:
        _default_embedding_service = EmbeddingService(model_type, model_name)
    return _default_embedding_service
