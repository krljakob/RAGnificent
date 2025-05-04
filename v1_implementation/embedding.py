#!/usr/bin/env python
"""Document embedding module for RAG implementation

This script converts text chunks into vector embeddings for semantic search
and stores them in Qdrant vector database. Multiple embedding models are
supported, with a fallback to simpler models when API keys are not available.
Includes robust error handling, input validation, and persistent storage.
"""
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, validator
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import EmbeddingConfig, QdrantConfig, load_config
from chunking import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables for API keys
load_dotenv()

# Ensure data directory exists
data_dir = Path(__file__).parent.parent / 'data'
cache_dir = data_dir / 'cache' / 'embeddings'
os.makedirs(data_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

# Qdrant errors and models
class QdrantError(Exception):
    """Base exception for Qdrant operations"""
    pass


class QdrantConnectionError(QdrantError):
    """Exception for Qdrant connection issues"""
    pass


class QdrantOperationError(QdrantError):
    """Exception for Qdrant operation failures"""
    pass


class EmbeddingOptions(BaseModel):
    """Options for embedding generation"""
    model_name: str = Field("all-MiniLM-L6-v2", description="Embedding model name")
    batch_size: int = Field(32, ge=1, le=512, description="Batch size for processing")
    device: str = Field("cpu", description="Device to use (cpu/cuda)")
    use_cache: bool = Field(True, description="Whether to use embedding cache")
    cache_dir: Optional[Path] = Field(None, description="Cache directory path")
    vector_size: int = Field(384, ge=1, description="Vector size for embeddings")
    normalize: bool = Field(True, description="Whether to normalize vectors")
    
    @validator('model_name')
    def validate_model_name(cls, v):
        """Basic validation for model name"""
        if not v or len(v) < 2:
            raise ValueError("Model name must be valid")
        return v


@lru_cache(maxsize=1)
def get_qdrant_client(config: Optional[QdrantConfig] = None) -> QdrantClient:
    """Initialize and return Qdrant client with connection retry.
    
    Args:
        config: Optional QdrantConfig instance
        
    Returns:
        Configured QdrantClient instance
        
    Raises:
        QdrantConnectionError: If connection fails
    """
    # Load config if not provided
    if config is None:
        try:
            app_config = load_config()
            config = app_config.qdrant
            logger.debug("Using application config for Qdrant connection")
        except Exception as e:
            logger.warning(f"Failed to load config, using default Qdrant settings: {e}")
            config = QdrantConfig()
    
    # Determine connection parameters
    host = config.host
    port = config.port
    is_memory = host == ":memory:"
    connection_args = {}
    
    # Configure client based on connection type
    try:
        if is_memory:
            logger.info("Using in-memory Qdrant database")
            return QdrantClient(location=":memory:")
        else:
            # HTTP URL or host/port connection
            if host.startswith("http://") or host.startswith("https://"):
                connection_args["url"] = host
                if config.https and not host.startswith("https://"):
                    logger.warning("HTTPS requested but URL doesn't use https scheme")
            else:
                connection_args["host"] = host
                connection_args["port"] = port
                connection_args["https"] = config.https
            
            # Add other connection parameters
            connection_args["api_key"] = config.api_key
            connection_args["prefer_grpc"] = config.prefer_grpc
            connection_args["timeout"] = config.timeout
            
            logger.info(f"Connecting to Qdrant at {host}" + (f":{port}" if "host" in connection_args else ""))
            
            # Try to connect with retry logic
            retries = 3
            last_error = None
            
            for attempt in range(1, retries + 1):
                try:
                    client = QdrantClient(**connection_args)
                    # Test connection with a simple operation
                    client.get_collections()
                    logger.info("Successfully connected to Qdrant")
                    return client
                except Exception as e:
                    last_error = e
                    logger.warning(f"Qdrant connection attempt {attempt}/{retries} failed: {str(e)}")
                    if attempt < retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        
            # All retries failed
            raise QdrantConnectionError(f"Failed to connect to Qdrant after {retries} attempts: {str(last_error)}")
                
    except Exception as e:
        if not isinstance(e, QdrantConnectionError):
            logger.error(f"Unexpected error connecting to Qdrant: {str(e)}")
            raise QdrantConnectionError(f"Qdrant connection error: {str(e)}") from e
        raise


def init_qdrant_collection(
    client: Optional[QdrantClient] = None,
    collection_name: Optional[str] = None,
    vector_size: int = 384,
    config: Optional[QdrantConfig] = None,
    recreate: bool = False
) -> Tuple[QdrantClient, str]:
    """Initialize Qdrant collection with proper configuration and error handling.

    Args:
        client: Qdrant client instance (or None to create new)
        collection_name: Name of the collection to create/check
        vector_size: Dimension of the embedding vectors
        config: Optional QdrantConfig to use
        recreate: Whether to recreate the collection if it exists

    Returns:
        Tuple of (client, collection_name)
        
    Raises:
        QdrantOperationError: If collection initialization fails
    """
    try:
        # Get config if not provided
        if config is None:
            try:
                app_config = load_config()
                config = app_config.qdrant
            except Exception as e:
                logger.warning(f"Failed to load config, using default Qdrant settings: {e}")
                config = QdrantConfig()
        
        # Use config values or defaults
        collection_name = collection_name or config.collection
        vector_size = vector_size or config.vector_size
        
        # Get client if not provided
        if client is None:
            client = get_qdrant_client(config)
            
        # Check for existing collections
        try:
            collections = client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            collection_exists = collection_name in collection_names
            
            # Handle collection creation or recreation
            if collection_exists:
                if recreate:
                    logger.info(f"Recreating Qdrant collection: {collection_name}")
                    client.delete_collection(collection_name=collection_name)
                    collection_exists = False
                else:
                    # Verify vector size of existing collection
                    try:
                        collection_info = client.get_collection(collection_name=collection_name)
                        existing_vector_size = collection_info.config.params.vectors.size
                        
                        if existing_vector_size != vector_size:
                            logger.warning(
                                f"Collection exists with different vector size: "
                                f"expected {vector_size}, got {existing_vector_size}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to verify existing collection vector size: {e}")
                    
                    logger.info(f"Using existing Qdrant collection: {collection_name}")
            
            # Create collection if needed
            if not collection_exists:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created new Qdrant collection: {collection_name}")
                
            return client, collection_name
            
        except UnexpectedResponse as e:
            logger.error(f"Unexpected response from Qdrant: {str(e)}")
            raise QdrantOperationError(f"Failed to initialize collection: {str(e)}") from e
            
    except Exception as e:
        if isinstance(e, QdrantError):
            raise
        logger.error(f"Error initializing Qdrant collection: {str(e)}")
        raise QdrantOperationError(f"Failed to initialize Qdrant collection: {str(e)}") from e

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
data_dir = Path(__file__).parent.parent / 'data'
os.makedirs(data_dir, exist_ok=True)


# Embedding exceptions
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
    """Get path for cached embeddings
    
    Args:
        model_name: Name of the embedding model
        text_hash: Hash of the text content
        
    Returns:
        Path to the cache file
    """
    # Create safe filename from model name
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    
    # Organize cache by model name
    model_cache_dir = cache_dir / safe_model_name
    os.makedirs(model_cache_dir, exist_ok=True)
    
    # Return path with hash as filename
    return model_cache_dir / f"{text_hash}.npy"


def compute_text_hash(text: str) -> str:
    """Compute deterministic hash for text content
    
    Args:
        text: Text to hash
        
    Returns:
        Hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_cached_embedding(model_name: str, text: str) -> Optional[np.ndarray]:
    """Get embedding from cache if available
    
    Args:
        model_name: Name of the embedding model
        text: Text content
        
    Returns:
        Cached embedding if available, None otherwise
    """
    try:
        text_hash = compute_text_hash(text)
        cache_path = get_embedding_cache_path(model_name, text_hash)
        return np.load(cache_path) if os.path.exists(cache_path) else None
    except Exception as e:
        logger.debug(f"Error retrieving cached embedding: {str(e)}")
        return None


def save_embedding_to_cache(model_name: str, text: str, embedding: np.ndarray) -> bool:
    """Save embedding to cache
    
    Args:
        model_name: Name of the embedding model
        text: Text content
        embedding: The embedding vector
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        text_hash = compute_text_hash(text)
        cache_path = get_embedding_cache_path(model_name, text_hash)
        np.save(cache_path, embedding)
        return True
    except Exception as e:
        logger.debug(f"Error saving embedding to cache: {str(e)}")
        return False


def tokenize_text(text: str) -> List[str]:
    """Simple tokenization function for text.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of tokens
    """
    # Simple tokenization by splitting on whitespace and removing punctuation
    return re.findall(r'\b\w+\b', text.lower())


def get_simple_count_embeddings(texts: List[str], max_features: int = 1000) -> np.ndarray:
    """Generate extremely simple word count embeddings as a last resort.
    
    This is the most basic fallback method when all other embedding approaches fail.
    It creates a simple bag-of-words representation with frequency counts.
    
    Args:
        texts: List of text strings to embed
        max_features: Maximum vocabulary size
        
    Returns:
        Numpy array of embeddings
    """
    if not texts:
        return np.array([])  # Return empty array for empty input
        
    try:
        logger.info(f"Generating simple count embeddings for {len(texts)} texts")
        start_time = time.time()
        
        # Build vocabulary from most common words
        all_words = []
        for text in texts:
            # Use the tokenize_text function for consistent tokenization
            all_words.extend(tokenize_text(text))
            
        # Count word frequencies
        from collections import Counter
        word_counts = Counter(all_words)
        
        # Take top words by frequency to form vocabulary
        vocab = [word for word, count in word_counts.most_common(max_features)]
        vocab_lookup = {word: i for i, word in enumerate(vocab)}
        
        # Create embeddings as simple count vectors
        embeddings = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if word in vocab_lookup:
                    embeddings[i, vocab_lookup[word]] += 1
                    
        # Basic L2 normalization to improve similarity comparisons
        from sklearn.preprocessing import normalize
        try:
            embeddings = normalize(embeddings, norm='l2', axis=1)
        except Exception:
            # If normalization fails, at least ensure no zero vectors
            for i in range(embeddings.shape[0]):
                if np.sum(embeddings[i]) == 0:
                    embeddings[i, 0] = 1.0  # Add at least one non-zero value
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated simple count embeddings in {elapsed_time:.2f}s with shape {embeddings.shape}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error generating simple count embeddings: {str(e)}")
        # Last resort: create random embeddings with consistent dimensions
        logger.warning("Creating random embeddings as absolute last resort")
        dim = min(100, max_features)
        random_embeddings = np.random.rand(len(texts), dim).astype(np.float32)
        # Normalize to unit length for better similarity comparison
        norms = np.linalg.norm(random_embeddings, axis=1, keepdims=True)
        random_embeddings = random_embeddings / np.maximum(norms, 1e-10)
        return random_embeddings
    except Exception as random_error:
        # At this point, we've exhausted all options
        raise EmbeddingError(f"All embedding methods failed and even random fallback failed: {str(random_error)}") from last_error


def embed_chunks(
    chunks: List[Dict[str, Any]],
    embedding_function: Optional[Callable] = None,
    chunk_key: str = 'text',
    collection_name: Optional[str] = None,
    store_vectors: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[EmbeddingConfig] = None
) -> List[Dict[str, Any]]:
    """Embed text chunks and store them in Qdrant vector database with enhanced reliability.

    Args:
        chunks: List of chunk dictionaries with text field
        embedding_function: Custom function to generate embeddings
        chunk_key: Key in chunk dictionaries containing the text to embed
        collection_name: Name of Qdrant collection to use
        store_vectors: Whether to store vectors in Qdrant
        output_path: Path to save embedded chunks JSON file
        config: Optional embedding configuration

    Returns:
        List of chunk dictionaries with added embedding field
        
    Raises:
        ValueError: If input validation fails
        EmbeddingError: If embedding generation fails
        QdrantError: If vector storage fails
    """
    # Validate inputs
    if not chunks:
        logger.warning("Empty chunks list provided to embed_chunks")
        return []
        
    logger.info(f"Embedding {len(chunks)} chunks")
    
    # Load config if not provided
    if config is None:
        try:
            app_config = load_config()
            config = app_config.embedding
            logger.debug("Using application config for embeddings")
        except Exception as e:
            logger.warning(f"Failed to load config, using default embedding settings: {e}")
            config = EmbeddingConfig()
    
    # Set up persistence paths
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = data_dir / f'embedded_chunks_{timestamp}.json'
        
    # Track timing for performance monitoring
    start_time = time.time()

    # Choose embedding function based on configuration
    if embedding_function is None:
        try:
            if config.model_type == EmbeddingModelType.OPENAI:
                embedding_function = lambda texts: get_openai_embeddings(
                    texts,
                    model=config.model_name,
                    batch_size=config.batch_size,
                    use_cache=config.use_cache
                )
                logger.info("Using OpenAI embeddings")
            elif config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMER:
                embedding_function = lambda texts: get_sentence_transformer_embeddings(
                    texts, 
                    model_name=config.model_name,
                    batch_size=config.batch_size,
                    device=config.device,
                    use_cache=config.use_cache
                )
                logger.info(f"Using Sentence Transformer embeddings with model {config.model_name}")
            elif config.model_type == EmbeddingModelType.TFIDF:
                embedding_function = lambda texts: get_tfidf_embeddings(
                    texts,
                    max_features=config.vector_size,
                    use_cache=config.use_cache
                )
                logger.info("Using TF-IDF embeddings")
            else:
                # Use the local embeddings function which tries multiple methods
                embedding_function = lambda texts: get_local_embeddings(texts, config=config)
                logger.info("Using automatic local embeddings selection")
        except Exception as e:
            logger.warning(f"Failed to set up preferred embedding function: {str(e)}")
            logger.info("Falling back to local embeddings")
            embedding_function = lambda texts: get_local_embeddings(texts, config=config)

    # Extract texts from chunks with validation
    texts = []
    for i, chunk in enumerate(chunks):
        text = chunk.get(chunk_key, "")
        if not text or not text.strip():
            logger.warning(f"Chunk {i} has empty text content, using placeholder.")
            text = f"[Empty chunk {i}]"
        texts.append(text)

    # Generate embeddings with proper error handling
    logger.info(f"Generating embeddings for {len(texts)} texts")
    try:
        embeddings = embedding_function(texts)
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Embedding function returned wrong number of embeddings: "
                f"got {len(embeddings)}, expected {len(texts)}"
            )
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
    except Exception as e:
        logger.error(f"Preferred embedding method failed: {str(e)}")
        logger.info("Attempting automatic fallback to local embeddings")
        try:
            embeddings = get_local_embeddings(texts, config=config)
            logger.info(f"Fallback successful, generated embeddings with shape {embeddings.shape}")
        except Exception as e2:
            error_msg = f"All embedding methods failed: {str(e)}; Fallback error: {str(e2)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e2

    # Add embeddings to chunks
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_copy = chunk.copy()
        # Add timestamp and metadata for tracking
        chunk_copy['embedding_timestamp'] = datetime.now().isoformat()
        chunk_copy['embedding_model'] = getattr(config, 'model_name', 'unknown')
        # Store the actual embedding vector
        chunk_copy['embedding'] = embeddings[i].tolist()
        embedded_chunks.append(chunk_copy)
        
    # Calculate embedding statistics for logging
    embedding_dim = len(embedded_chunks[0]['embedding']) if embedded_chunks else 0
    elapsed_time = time.time() - start_time
    embedding_rate = len(chunks) / max(0.001, elapsed_time)
    logger.info(f"Created {len(embedded_chunks)} embeddings of dimension {embedding_dim} in {elapsed_time:.2f}s ({embedding_rate:.1f} chunks/sec)")

    # Save embedded chunks to file
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            # Add metadata for the embeddings
            output_data = {
                'chunks': embedded_chunks,
                'metadata': {
                    'count': len(embedded_chunks),
                    'embedding_dim': embedding_dim,
                    'model': getattr(config, 'model_name', 'unknown'),
                    'created_at': datetime.now().isoformat()
                }
            }
            # Save with compact representation for embeddings to save space
            json.dump(output_data, f, default=lambda x: x if not isinstance(x, np.ndarray) else x.tolist())
        logger.info(f"Saved embedded chunks to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save embedded chunks: {str(e)}")

    # Store in Qdrant if requested
    if store_vectors:
        try:
            # Get Qdrant client and collection
            client = get_qdrant_client()
            
            # Initialize collection with correct embedding dimension
            actual_collection_name = init_qdrant_collection(
                client=client,
                collection_name=collection_name,
                embedding_dim=embedding_dim
            )

            # Prepare points for Qdrant
            points = []
            for i, chunk in enumerate(embedded_chunks):
                # Create a unique ID for each chunk (use existing or generate)
                chunk_id = chunk.get('chunk_id', f"chunk_{uuid4().hex}")
                
                # Extract metadata (excluding large fields)
                metadata = {}
                for k, v in chunk.items():
                    # Skip embedding vector and very large text fields
                    if k == 'embedding' or (isinstance(v, str) and len(v) > 8000):
                        continue
                    # For text over 1KB, truncate it
                    if isinstance(v, str) and len(v) > 1000:
                        metadata[k] = v[:1000] + "... [truncated]"
                    else:
                        metadata[k] = v
                
                # Ensure essential fields are present
                if chunk_key not in metadata and chunk_key in chunk:
                    # Include at least a summary of the text
                    text = chunk[chunk_key]
                    metadata[chunk_key] = text[:1000] + "..." if len(text) > 1000 else text
                
                # Create Qdrant point
                point = models.PointStruct(
                    id=chunk_id,
                    vector=chunk['embedding'],
                    payload=metadata
                )
                points.append(point)

            # Upsert in batches to avoid timeouts
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                client.upsert(
                    collection_name=actual_collection_name,
                    points=batch,
                    wait=True
                )
                logger.info(f"Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} in Qdrant")

            logger.info(f"Successfully stored {len(points)} points in Qdrant collection '{actual_collection_name}'")
        except Exception as e:
            logger.error(f"Failed to store embeddings in Qdrant: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning("Continuing with local embeddings only")
            # If Qdrant storage was critical, we might want to raise this error
            if isinstance(e, QdrantError):
                raise

    return embedded_chunks


def search_qdrant(
    query: str,
    embedding_function: Optional[Callable] = None,
    collection_name: Optional[str] = None,
    limit: int = 5,
    score_threshold: float = 0.7,
    filter_condition: Optional[Dict] = None,
    config: Optional[EmbeddingConfig] = None,
    with_payload: bool = True,
    with_vectors: bool = False
) -> List[Dict[str, Any]]:
    """Search Qdrant collection for similar documents with enhanced reliability.
    
    This function implements semantic search by embedding the query and finding
    similar vectors in the Qdrant vector database.

    Args:
        query: Search query text
        embedding_function: Function to generate query embedding
        collection_name: Specific collection to search in (uses default if None)
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score (0-1) for results
        filter_condition: Optional Qdrant filter condition for refined search
        config: Optional embedding configuration
        with_payload: Whether to return document payload
        with_vectors: Whether to return document vectors

    Returns:
        List of matching documents with scores and metadata
        
    Raises:
        QdrantError: If Qdrant search fails
        EmbeddingError: If embedding generation fails
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to search_qdrant")
        return []
        
    # Load config if not provided
    if config is None:
        try:
            app_config = load_config()
            config = app_config.embedding
            logger.debug("Using application config for search")
        except Exception as e:
            logger.warning(f"Failed to load config, using default embedding settings: {e}")
            config = EmbeddingConfig()
    
    # Initialize Qdrant client
    try:
        client = get_qdrant_client()
    except QdrantError as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise
        
    # Use the appropriate collection name
    if collection_name is None:
        try:
            app_config = load_config()
            collection_name = app_config.qdrant.collection_name
        except Exception:
            collection_name = QDRANT_COLLECTION_NAME
            
    # Track timing for performance monitoring
    start_time = time.time()

    # Choose embedding function based on configuration if not provided
    if embedding_function is None:
        try:
            # Try to use the same embedding method as used for documents
            if config.model_type == EmbeddingModelType.OPENAI:
                embedding_function = lambda texts: get_openai_embeddings(
                    texts,
                    model=config.model_name,
                    batch_size=1,  # For queries, batch size is always 1
                    use_cache=False  # Don't cache query embeddings
                )
                logger.info("Using OpenAI embeddings for search query")
            elif config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMER:
                embedding_function = lambda texts: get_sentence_transformer_embeddings(
                    texts, 
                    model_name=config.model_name,
                    batch_size=1,
                    device=config.device,
                    use_cache=False
                )
                logger.info(f"Using Sentence Transformer embeddings with model {config.model_name} for search query")
            else:
                # For other models like TF-IDF, we need to ensure we use the same model as indexing
                embedding_function = lambda texts: get_local_embeddings(texts, config=config)
                logger.info("Using local embeddings for search query")
        except Exception as e:
            logger.warning(f"Failed to set up preferred embedding function: {e}")
            embedding_function = lambda texts: get_local_embeddings(texts, config=config)

    try:
        # Generate query embedding with error handling
        logger.info(f"Generating embedding for query: '{query[:50]}...' if len(query) > 50 else query")
        try:
            query_embedding = embedding_function([query])[0]
            
            # Convert embedding to list if it's a numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
                
            logger.info(f"Generated query embedding with dimension {len(query_embedding)}")
        except Exception as embed_error:
            logger.error(f"Failed to generate embedding for query: {embed_error}")
            raise EmbeddingError(f"Query embedding failed: {embed_error}") from embed_error

        # Validate search parameters
        if limit <= 0:
            logger.warning(f"Invalid limit {limit}, using default 5")
            limit = 5
            
        if not (0 <= score_threshold <= 1):
            logger.warning(f"Invalid score_threshold {score_threshold}, using default 0.7")
            score_threshold = 0.7

        # Search Qdrant with error handling
        try:
            logger.info(f"Searching collection '{collection_name}' with limit={limit}, threshold={score_threshold}")
            search_params = {
                "collection_name": collection_name,
                "query_vector": query_embedding,
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": with_payload,
                "with_vectors": with_vectors
            }
            
            # Add filter if provided
            if filter_condition:
                search_params["query_filter"] = filter_condition
                
            results = client.search(**search_params)
            logger.info(f"Search returned {len(results)} results")
        except Exception as search_error:
            logger.error(f"Qdrant search failed: {search_error}")
            raise QdrantError(f"Vector search failed: {search_error}") from search_error

        # Calculate search time for logging
        elapsed_time = time.time() - start_time
        logger.info(f"Search completed in {elapsed_time:.2f}s")

        # Format results into a user-friendly structure
        matches = []
        for i, result in enumerate(results):
            try:
                # Extract core information
                match = {
                    'score': float(result.score),  # Ensure score is a Python float
                    'id': result.id,
                    'position': i + 1  # 1-based position in results
                }
                
                # Add payload if available
                if hasattr(result, 'payload') and result.payload:
                    # Extract text content
                    if 'text' in result.payload:
                        match['text'] = result.payload['text']
                    
                    # Add all payload fields as metadata
                    match['metadata'] = {k: v for k, v in result.payload.items() if k != 'text'}
                    
                # Add vector if requested
                if with_vectors and hasattr(result, 'vector') and result.vector:
                    match['vector'] = result.vector
                    
                matches.append(match)
            except Exception as format_error:
                logger.warning(f"Error formatting result {i}: {format_error}")
                # Continue with other results even if one fails

        return matches

    except (QdrantError, EmbeddingError):
        # Re-raise specific errors
        raise
    except Exception as e:
        logger.error(f"Unexpected error during vector search: {e}")
        logger.error(traceback.format_exc())
        return []


if __name__ == "__main__":
    # Load chunked documents
    chunks_path = data_dir / 'document_chunks.json'
    if not chunks_path.exists():
        logger.error(f"Document chunks not found at {chunks_path}")
        logger.info("Run 2_chunking.py first to create document chunks")
        sys.exit(1)

    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Embed chunks
    embedded_chunks = embed_chunks(chunks)

    # Print summary
    num_with_embeddings = sum('embedding' in chunk for chunk in embedded_chunks)

    if num_with_embeddings > 0:
        sample_chunk = next(chunk for chunk in embedded_chunks if 'embedding' in chunk)
        embedding_dim = len(sample_chunk['embedding'])

        # Print first few dimensions of first embedding
