"""
Vector store module for RAGnificent.
Provides interface to Qdrant or other vector databases for storing and retrieving embeddings.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

# Use v1_implementation's vector store functionality
from v1_implementation.vector_store import initialize_collection
from v1_implementation.search import search_chunks as v1_search_chunks
from v1_implementation.config import get_config

logger = logging.getLogger(__name__)

class VectorStore:
    """Interface to vector database for storing and retrieving embeddings."""

    def __init__(
        self, 
        collection_name: Optional[str] = None,
        embedding_dim: Optional[int] = None
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the collection to use
            embedding_dim: Dimension of embeddings
        """
        self.config = get_config()
        self.collection_name = collection_name or self.config.get("vector_store", {}).get(
            "collection_name", "ragnificent"
        )
        self.embedding_dim = embedding_dim or self.config.get("embedding", {}).get(
            "dimension", 384  # Default for bge-small
        )
        
        try:
            # Initialize the vector store collection
            self.client = initialize_collection(
                collection_name=self.collection_name,
                embedding_dim=self.embedding_dim
            )
            logger.info(f"Initialized vector store with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def store_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Store chunks with embeddings in the vector database.

        Args:
            chunks: List of chunks with embeddings

        Returns:
            Success flag
        """
        try:
            # Filter chunks that have embeddings
            valid_chunks = [chunk for chunk in chunks if "embedding" in chunk]
            if not valid_chunks:
                logger.warning("No chunks with embeddings found to store")
                return False
                
            # Prepare points for insertion
            points = []
            for chunk in valid_chunks:
                # Ensure ID is string for Qdrant
                chunk_id = str(chunk.get("id", ""))
                if not chunk_id:
                    continue
                    
                payload = {k: v for k, v in chunk.items() if k != "embedding"}
                
                points.append({
                    "id": chunk_id,
                    "vector": chunk["embedding"],
                    "payload": payload
                })
            
            if not points:
                logger.warning("No valid points to insert")
                return False
                
            # Insert to collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully stored {len(points)} chunks in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunks in vector store: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        limit: int = 5, 
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.

        Args:
            query: Query text
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            List of matching chunks with scores
        """
        try:
            # Use v1_implementation's search function
            results = v1_search_chunks(
                query=query, 
                collection_name=self.collection_name,
                limit=limit,
                score_threshold=threshold
            )
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []

# Singleton instance for easy access
default_vector_store = None

def get_vector_store(
    collection_name: Optional[str] = None,
    embedding_dim: Optional[int] = None
) -> VectorStore:
    """
    Get or create the default vector store.

    Args:
        collection_name: Name of the collection to use
        embedding_dim: Dimension of embeddings

    Returns:
        The vector store instance
    """
    global default_vector_store
    if default_vector_store is None:
        default_vector_store = VectorStore(collection_name, embedding_dim)
    return default_vector_store
