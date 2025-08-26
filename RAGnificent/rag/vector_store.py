"""
Vector store module for RAGnificent.

Provides interface to vector databases (primarily Qdrant) for storing and retrieving embeddings.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Use relative imports for internal modules
try:
    from ..core.config import get_config
    from ..rag.embedding import embed_text, get_embedding_model
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import get_config
    from rag.embedding import embed_text, get_embedding_model

import numpy as np

logger = logging.getLogger(__name__)


class QdrantError(Exception):
    """Base exception for Qdrant operations"""

    pass


class QdrantConnectionError(QdrantError):
    """Exception for Qdrant connection issues"""

    pass


class QdrantOperationError(QdrantError):
    """Exception for Qdrant operation failures"""

    pass


def get_qdrant_client(config_override: Optional[Dict[str, Any]] = None) -> Any:
    """
    Initialize and return Qdrant client with connection retry.

    Args:
        config_override: Optional configuration override

    Returns:
        Configured QdrantClient instance

    Raises:
        QdrantConnectionError: If connection fails
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http.exceptions import UnexpectedResponse

        config = get_config()

        # Apply config override if provided
        if config_override:
            host = config_override.get("host", config.qdrant.host)
            port = config_override.get("port", config.qdrant.port)
            api_key = config_override.get("api_key", config.qdrant.api_key)
            https = config_override.get("https", config.qdrant.https)
            timeout = config_override.get("timeout", config.qdrant.timeout)
            prefer_grpc = config_override.get("prefer_grpc", config.qdrant.prefer_grpc)
        else:
            host = config.qdrant.host
            port = config.qdrant.port
            api_key = config.qdrant.api_key
            https = config.qdrant.https
            timeout = config.qdrant.timeout
            prefer_grpc = config.qdrant.prefer_grpc

        # Handle in-memory database
        if host.lower() == ":memory:":
            logger.info("Using in-memory Qdrant database")
            return QdrantClient(":memory:")

        # Connect with retry logic
        max_retries = 3
        retry_count = 0
        last_exception = None

        while retry_count < max_retries:
            try:
                client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    https=https,
                    timeout=timeout,
                    prefer_grpc=prefer_grpc,
                )

                # Test connection
                client.get_collections()
                logger.info(f"Successfully connected to Qdrant at {host}:{port}")
                return client

            except UnexpectedResponse as e:
                retry_count += 1
                last_exception = e
                wait_time = 2**retry_count  # Exponential backoff
                logger.warning(
                    f"Qdrant connection attempt {retry_count}/{max_retries} failed: {e}. Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise QdrantConnectionError(f"Failed to connect to Qdrant: {e}") from e

        # If we got here, all retries failed
        logger.error(
            f"Failed to connect to Qdrant after {max_retries} retries: {last_exception}"
        )
        raise QdrantConnectionError(
            f"Failed to connect to Qdrant after {max_retries} retries: {last_exception}"
        )

    except ImportError as exc:
        logger.error(
            "Qdrant client not installed. Install with: uv pip install qdrant-client"
        )
        raise QdrantConnectionError("Qdrant client not installed") from exc


def initialize_collection(
    collection_name: Optional[str] = None,
    vector_size: Optional[int] = None,
    client: Optional[Any] = None,
    distance: str = "Cosine",
    recreate: bool = False,
) -> Tuple[Any, str]:
    """
    Initialize Qdrant collection with proper configuration and error handling.

    Args:
        collection_name: Name of the collection to create/check
        vector_size: Dimension of the embedding vectors
        client: Qdrant client instance (or None to create new)
        distance: Distance function to use (Cosine, Euclid, Dot)
        recreate: Whether to recreate the collection if it exists

    Returns:
        Tuple of (client, collection_name)

    Raises:
        QdrantOperationError: If collection initialization fails
    """
    try:
        from qdrant_client.http import models
        from qdrant_client.http.exceptions import UnexpectedResponse

        config = get_config()

        # Get client if not provided
        if client is None:
            client = get_qdrant_client()

        # Use config values if not specified
        if collection_name is None:
            collection_name = config.qdrant.collection

        if vector_size is None:
            vector_size = config.embedding.dimension

        # Check if collection exists
        try:
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name in collection_names:
                if recreate:
                    logger.info(f"Recreating collection '{collection_name}'")
                    client.delete_collection(collection_name=collection_name)
                else:
                    logger.info(f"Collection '{collection_name}' already exists")
                    return client, collection_name

        except UnexpectedResponse as e:
            logger.warning(f"Error checking collections, will try to create: {e}")

        # Create new collection
        logger.info(
            f"Creating collection '{collection_name}' with vector size {vector_size}"
        )
        # Map distance string to Qdrant enum
        distance_map = {
            "Cosine": models.Distance.COSINE,
            "cosine": models.Distance.COSINE,
            "Euclid": models.Distance.EUCLID,
            "euclid": models.Distance.EUCLID,
            "Dot": models.Distance.DOT,
            "dot": models.Distance.DOT,
        }
        dist_enum = distance_map.get(distance, models.Distance.COSINE)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=dist_enum),
        )

        logger.info(f"Successfully created collection '{collection_name}'")
        return client, collection_name

    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection: {e}")
        raise QdrantOperationError(
            f"Failed to initialize Qdrant collection: {e}"
        ) from e


class VectorStore:
    """Interface to vector database for storing and retrieving embeddings."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        client: Optional[Any] = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the collection to use
            vector_size: Dimension of embeddings
            client: Qdrant client instance (or None to create new)
        """
        try:
            config = get_config()
            self.collection_name = collection_name or config.qdrant.collection
            self.vector_size = vector_size or config.embedding.dimension

            # Initialize Qdrant client and collection
            self.client, self.collection_name = initialize_collection(
                collection_name=self.collection_name,
                vector_size=self.vector_size,
                client=client,
            )

            logger.info(
                f"Initialized vector store with collection: {self.collection_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def store_documents(
        self,
        documents: List[Dict[str, Any]],
        embedding_field: str = "embedding",
        id_field: str = "id",
        batch_size: int = 100,
    ) -> bool:
        """
        Store documents with embeddings in the vector database.

        Args:
            documents: List of documents with embeddings
            embedding_field: Field name containing the embedding
            id_field: Field name containing the document ID
            batch_size: Size of batches for insertion

        Returns:
            Success flag
        """
        try:
            from qdrant_client.http import models

            # Ensure all documents have embeddings
            valid_docs = [doc for doc in documents if embedding_field in doc]

            if not valid_docs:
                logger.warning("No documents with embeddings found to store")
                return False

            total_docs = len(valid_docs)
            logger.info(f"Storing {total_docs} documents in vector store")

            # Check the actual dimension of the first embedding and recreate collection if necessary
            sample_embedding = valid_docs[0][embedding_field]
            actual_dim = len(sample_embedding)
            if actual_dim != self.vector_size:
                logger.warning(
                    f"Embedding dimension mismatch. Expected {self.vector_size}, got {actual_dim}. Recreating collection."
                )
                # Reinitialize the collection with the actual dimension
                self.client, self.collection_name = initialize_collection(
                    collection_name=self.collection_name,
                    vector_size=actual_dim,
                    client=self.client,
                    recreate=True,
                )
                self.vector_size = actual_dim

            # Process in batches for efficiency
            for i in range(0, total_docs, batch_size):
                batch = valid_docs[i : i + batch_size]

                # Prepare points for insertion
                points = []
                for doc in batch:
                    # Ensure ID is string for Qdrant
                    doc_id = str(doc.get(id_field, "")) or str(
                        hash(frozenset(doc.items()))
                    )

                    # Prepare payload (all fields except embedding)
                    payload = {k: v for k, v in doc.items() if k != embedding_field}

                    # Get embedding as list
                    embedding = doc[embedding_field]
                    if isinstance(embedding, np.ndarray):
                        embedding = embedding.tolist()

                    points.append(
                        models.PointStruct(id=doc_id, vector=embedding, payload=payload)
                    )

                # Insert batch to collection
                self.client.upsert(collection_name=self.collection_name, points=points)

                logger.info(
                    f"Stored batch of {len(batch)} documents ({i+len(batch)}/{total_docs})"
                )

            logger.info(f"Successfully stored {total_docs} documents in vector store")
            return True

        except Exception as e:
            logger.error(f"Error storing documents in vector store: {e}")
            return False

    def search(
        self,
        query_text: str = None,
        query_vector: Optional[List[float]] = None,
        limit: int = 5,
        threshold: float = 0.7,
        filter_condition: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query_text: Query text (ignored if query_vector provided)
            query_vector: Query vector (generated from query_text if not provided)
            limit: Maximum number of results
            threshold: Similarity threshold (0-1)
            filter_condition: Additional filter conditions for search
            with_payload: Whether to include document payload in results
            with_vectors: Whether to include vectors in results

        Returns:
            List of matching documents with scores
        """
        try:
            from qdrant_client.http import models

            # Get embedding vector from text if not provided
            if query_vector is None and query_text is not None:
                embedding_model = get_embedding_model()
                query_vector = embed_text(query_text, embedding_model)

                # Convert numpy array to list if needed
                if isinstance(query_vector, np.ndarray):
                    query_vector = query_vector.tolist()

            if query_vector is None:
                raise ValueError("Either query_text or query_vector must be provided")

            search_filter = (
                models.Filter(**filter_condition) if filter_condition else None
            )
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=threshold,
                query_filter=search_filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )

            # Convert result to dictionaries
            results = []
            for scored_point in search_result:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                }

                # Add payload if included
                if with_payload and scored_point.payload:
                    result["payload"] = scored_point.payload

                # Add vector if included
                if with_vectors and scored_point.vector:
                    result["vector"] = scored_point.vector

                results.append(result)

            logger.info(f"Found {len(results)} results for query")
            return results

        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

    def delete_by_ids(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            Success flag
        """
        try:
            from qdrant_client.http import models

            if not ids:
                logger.warning("No IDs provided for deletion")
                return False

            # Convert all IDs to strings
            string_ids = [str(id) for id in ids]

            # Delete points by IDs
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=string_ids),
            )

            logger.info(f"Successfully deleted {len(ids)} documents from vector store")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {e}")
            return False

    def delete_by_filter(self, filter_condition: Dict[str, Any]) -> bool:
        """
        Delete documents by filter condition.

        Args:
            filter_condition: Filter condition for documents to delete

        Returns:
            Success flag
        """
        try:
            from qdrant_client.http import models

            if not filter_condition:
                logger.warning("No filter condition provided for deletion")
                return False

            # Create filter
            filter_obj = models.Filter(**filter_condition)

            # Delete points by filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=filter_obj),
            )

            logger.info(
                "Successfully deleted documents matching filter from vector store"
            )
            return True

        except Exception as e:
            logger.error(f"Error deleting documents from vector store by filter: {e}")
            return False

    def count_documents(self, filter_condition: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents in the collection.

        Args:
            filter_condition: Optional filter condition

        Returns:
            Number of documents
        """
        try:
            from qdrant_client.http import models

            filter_obj = models.Filter(**filter_condition) if filter_condition else None
            # Count documents
            count = self.client.count(
                collection_name=self.collection_name, count_filter=filter_obj
            )

            return count.count

        except Exception as e:
            logger.error(f"Error counting documents in vector store: {e}")
            return 0


# Singleton instance for easy access
_default_vector_store = None


def get_vector_store(
    collection_name: Optional[str] = None, vector_size: Optional[int] = None
) -> VectorStore:
    """
    Get or create the default vector store.

    Args:
        collection_name: Name of the collection to use
        vector_size: Dimension of embeddings

    Returns:
        The vector store instance
    """
    global _default_vector_store
    if _default_vector_store is None:
        _default_vector_store = VectorStore(collection_name, vector_size)
    return _default_vector_store
