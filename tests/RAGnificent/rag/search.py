"""
Search module for RAGnificent.

Provides semantic search capabilities over embedded content with customizable
retrieval strategies, filtering, and reranking.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Use relative imports for internal modules
# Import fix applied
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.config import get_config
from rag.embedding import embed_text, get_embedding_model
from rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class SearchResult:
    """Structured search result with metadata."""

    def __init__(
        self,
        content: str,
        score: float,
        metadata: Dict[str, Any],
        document_id: str,
        source_url: Optional[str] = None,
    ):
        """
        Initialize search result.

        Args:
            content: The text content
            score: Relevance score (0-1)
            metadata: Additional metadata
            document_id: Unique document identifier
            source_url: Source URL if available
        """
        self.content = content
        self.score = score
        self.metadata = metadata
        self.document_id = document_id
        self.source_url = source_url

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "source_url": self.source_url,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create from dictionary."""
        return cls(
            content=data.get("content", ""),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {}),
            document_id=data.get("document_id", ""),
            source_url=data.get("source_url"),
        )


class SemanticSearch:
    """Semantic search over embeddings."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        vector_size: Optional[int] = None,
    ):
        """
        Initialize the semantic search.

        Args:
            collection_name: Name of the collection to search
            model_type: Type of embedding model
            model_name: Name of the embedding model to use
            vector_size: Vector dimension
        """
        config = get_config()
        self.collection_name = collection_name or config.qdrant.collection

        # Initialize embedding model
        self.embedding_model = get_embedding_model(model_type, model_name)

        # Initialize vector store
        self.vector_store = get_vector_store(self.collection_name, vector_size)

        # Cache for recent searches
        self._query_cache = {}
        self._use_cache = config.search.enable_caching
        self._cache_ttl = config.search.cache_ttl

        logger.info(
            f"Initialized semantic search with collection: {self.collection_name}"
        )

    def clear_cache(self):
        """Clear the search cache."""
        self._query_cache = {}

    def _check_cache(self, query: str, limit: int) -> Optional[List[SearchResult]]:
        """Check if query is cached."""
        if not self._use_cache:
            return None

        key = f"{query}:{limit}"
        if key in self._query_cache:
            entry = self._query_cache[key]
            # Check if entry is still valid
            if time.time() - entry["timestamp"] < self._cache_ttl:
                logger.info(f"Using cached search results for: {query}")
                return entry["results"]

            # Remove expired entry
            del self._query_cache[key]

        return None

    def _cache_results(self, query: str, limit: int, results: List[SearchResult]):
        """Cache search results."""
        if not self._use_cache:
            return

        key = f"{query}:{limit}"
        self._query_cache[key] = {"results": results, "timestamp": time.time()}

        # Clean up old entries
        if len(self._query_cache) > 100:  # Limit cache size
            now = time.time()
            self._query_cache = {
                k: v
                for k, v in self._query_cache.items()
                if now - v["timestamp"] < self._cache_ttl
            }

    def search(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None,
        rerank: bool = False,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        """
        Search for chunks similar to the query.

        Args:
            query: Query text
            limit: Maximum number of results
            threshold: Similarity threshold
            filter_conditions: Additional filter conditions
            rerank: Whether to rerank results for improved relevance
            include_vectors: Whether to include vectors in results

        Returns:
            List of search results
        """
        # Check cache first
        cached_results = self._check_cache(query, limit)
        if cached_results is not None:
            return cached_results

        try:
            # Embed query
            query_embedding = embed_text(query, self.embedding_model)

            # Convert numpy array to list if needed
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Search vector store
            raw_results = self.vector_store.search(
                query_vector=query_embedding,
                limit=limit * 2 if rerank else limit,  # Get more results if reranking
                threshold=threshold,
                filter_condition=filter_conditions,
                with_payload=True,
                with_vectors=include_vectors,
            )

            if not raw_results:
                logger.info(f"No results found for query: {query}")
                return []

            # Convert to SearchResult objects
            search_results = []
            for result in raw_results:
                payload = result.get("payload", {})

                # Get content
                content = payload.get("content", "")
                if not content and "text" in payload:
                    content = payload["text"]

                # Get metadata
                metadata = {
                    k: v for k, v in payload.items() if k not in ["content", "text"]
                }

                # Get source URL
                source_url = payload.get("source_url", None)
                if not source_url and "url" in payload:
                    source_url = payload["url"]

                search_results.append(
                    SearchResult(
                        content=content,
                        score=result.get("score", 0.0),
                        metadata=metadata,
                        document_id=result.get("id", ""),
                        source_url=source_url,
                    )
                )

            # Rerank results if requested
            if rerank and len(search_results) > limit:
                search_results = self._rerank_results(query, search_results, limit)

            # Limit results
            search_results = search_results[:limit]

            # Cache results
            self._cache_results(query, limit, search_results)

            logger.info(f"Found {len(search_results)} results for query: {query}")
            return search_results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _rerank_results(
        self, query: str, results: List[SearchResult], limit: int
    ) -> List[SearchResult]:
        """
        Rerank search results using more sophisticated relevance metrics.

        Args:
            query: Original query text
            results: Initial search results
            limit: Maximum number of results to return

        Returns:
            Reranked search results
        """
        try:
            # This is a simple reranking based on additional query terms matching
            # For more sophisticated reranking, we could use a cross-encoder model

            query_terms = set(query.lower().split())
            reranked = []

            for result in results:
                # Original score from vector similarity
                base_score = result.score

                # Calculate term overlap for boosting
                content_terms = set(result.content.lower().split())
                term_overlap = (
                    len(query_terms.intersection(content_terms)) / len(query_terms)
                    if query_terms
                    else 0
                )

                # Calculate content length penalty (prefer shorter, concise answers)
                length = len(result.content.split())
                length_factor = min(
                    1.0, 200 / max(length, 1)
                )  # Penalize very long content

                # Combine factors (weighted)
                adjusted_score = (
                    (base_score * 0.7) + (term_overlap * 0.2) + (length_factor * 0.1)
                )

                # Create updated result with adjusted score
                updated_result = SearchResult(
                    content=result.content,
                    score=adjusted_score,
                    metadata=result.metadata,
                    document_id=result.document_id,
                    source_url=result.source_url,
                )
                reranked.append(updated_result)

            # Sort by adjusted score
            reranked.sort(key=lambda x: x.score, reverse=True)
            return reranked[:limit]

        except Exception as e:
            logger.error(f"Error in result reranking: {e}")
            return results[:limit]  # Return original results if reranking fails


# Default search instance
_default_search = None


def get_search(
    collection_name: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    vector_size: Optional[int] = None,
) -> SemanticSearch:
    """
    Get or create the default semantic search.

    Args:
        collection_name: Name of the collection to search
        model_type: Type of embedding model
        model_name: Name of the embedding model to use
        vector_size: Vector dimension

    Returns:
        The semantic search instance
    """
    global _default_search
    if _default_search is None:
        _default_search = SemanticSearch(
            collection_name, model_type, model_name, vector_size
        )
    return _default_search


def search(
    query: str,
    limit: int = 5,
    threshold: float = 0.7,
    collection_name: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    filter_conditions: Optional[Dict[str, Any]] = None,
    rerank: bool = False,
    include_vectors: bool = False,
) -> List[Dict[str, Any]]:
    """
    Convenience function for searching.

    Args:
        query: Query text
        limit: Maximum number of results
        threshold: Similarity threshold
        collection_name: Name of the collection to search
        model_type: Type of embedding model
        model_name: Name of the embedding model to use
        filter_conditions: Additional filter conditions
        rerank: Whether to rerank results
        include_vectors: Whether to include vectors in results

    Returns:
        List of search results as dictionaries
    """
    search_instance = get_search(collection_name, model_type, model_name)
    results = search_instance.search(
        query, limit, threshold, filter_conditions, rerank, include_vectors
    )
    return [result.to_dict() for result in results]
