"""
Search module for RAGnificent.
Provides semantic search capabilities over embedded content.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from RAGnificent.rag.embedding import get_embedding_service
from RAGnificent.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)

class SemanticSearch:
    """Semantic search over embeddings."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the semantic search.

        Args:
            collection_name: Name of the collection to search
            model_name: Name of the embedding model to use
        """
        self.embedding_service = get_embedding_service(model_name)
        self.vector_store = get_vector_store(collection_name)
        
    def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks similar to the query.

        Args:
            query: Query text
            limit: Maximum number of results
            threshold: Similarity threshold
            filter_conditions: Additional filter conditions

        Returns:
            List of matching chunks with scores
        """
        try:
            # Use the vector store's search functionality
            results = self.vector_store.search(
                query=query,
                limit=limit,
                threshold=threshold
            )
            
            # Apply additional filtering if needed
            if filter_conditions and results:
                filtered_results = []
                for result in results:
                    payload = result.get("payload", {})
                    match = True
                    for key, value in filter_conditions.items():
                        if key in payload and payload[key] != value:
                            match = False
                            break
                    if match:
                        filtered_results.append(result)
                return filtered_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

# Default search instance
default_search = None

def get_search(
    collection_name: Optional[str] = None,
    model_name: Optional[str] = None
) -> SemanticSearch:
    """
    Get or create the default semantic search.

    Args:
        collection_name: Name of the collection to search
        model_name: Name of the embedding model to use

    Returns:
        The semantic search instance
    """
    global default_search
    if default_search is None:
        default_search = SemanticSearch(collection_name, model_name)
    return default_search
    
def search(
    query: str,
    limit: int = 5,
    threshold: float = 0.7,
    collection_name: Optional[str] = None,
    model_name: Optional[str] = None,
    filter_conditions: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function for searching.

    Args:
        query: Query text
        limit: Maximum number of results
        threshold: Similarity threshold
        collection_name: Name of the collection to search
        model_name: Name of the embedding model to use
        filter_conditions: Additional filter conditions

    Returns:
        List of matching chunks with scores
    """
    search = get_search(collection_name, model_name)
    return search.search(query, limit, threshold, filter_conditions)
