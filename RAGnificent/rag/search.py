"""
Search module for RAGnificent.

Provides semantic search capabilities over embedded content with customizable
retrieval strategies, filtering, and reranking.
"""

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Use relative imports for internal modules
try:
    from ..core.config import get_config
    from ..rag.embedding import embed_text, get_embedding_model
    from ..rag.vector_store import get_vector_store
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import get_config
    from rag.embedding import embed_text, get_embedding_model
    from rag.vector_store import get_vector_store

import numpy as np

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


class CrossEncoderReranker:
    """Cross-encoder model for semantic reranking of search results."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the cross-encoder model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Use GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
            logger.info(f"Loaded cross-encoder model: {self.model_name} on {self.device}")

        except ImportError as e:
            logger.error(
                "Transformers package not installed. Please install with: uv pip install transformers torch"
            )
            raise ImportError("Transformers package required for cross-encoder reranking") from e

        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {e}")
            raise RuntimeError(f"Failed to load cross-encoder model: {e}") from e

    def rerank(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: int = 5,
        batch_size: int = 16
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder model.

        Args:
            query: Original search query
            results: List of search results to rerank
            top_k: Number of top results to return
            batch_size: Batch size for model inference

        Returns:
            Reranked search results
        """
        if not results:
            return results

        try:
            import torch

            # Prepare query-document pairs
            pairs = [(query, result.content) for result in results]
            
            # Process in batches for efficiency
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # Tokenize batch
                queries = [pair[0] for pair in batch_pairs]
                documents = [pair[1] for pair in batch_pairs]
                
                encodings = self.tokenizer(
                    queries, 
                    documents,
                    padding=True, 
                    truncation=True,
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Get relevance scores
                with torch.no_grad():
                    outputs = self.model(**encodings)
                    scores = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
                    
                    # Handle single item case
                    if scores.ndim == 0:
                        scores = [float(scores)]
                    else:
                        scores = scores.tolist()
                    
                all_scores.extend(scores)

            # Update result scores and sort
            reranked_results = []
            for result, score in zip(results, all_scores):
                # Create new result with updated score
                reranked_result = SearchResult(
                    content=result.content,
                    score=float(score),
                    metadata={**result.metadata, "original_score": result.score},
                    document_id=result.document_id,
                    source_url=result.source_url,
                )
                reranked_results.append(reranked_result)

            # Sort by cross-encoder score
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Reranked {len(results)} results using cross-encoder")
            return reranked_results[:top_k]

        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            # Fallback to original results
            return results[:top_k]

    def batch_rerank(
        self, 
        query_result_pairs: List[Tuple[str, List[SearchResult]]], 
        top_k: int = 5
    ) -> List[List[SearchResult]]:
        """
        Rerank multiple query-result pairs in batch for efficiency.

        Args:
            query_result_pairs: List of (query, results) tuples
            top_k: Number of top results to return for each query

        Returns:
            List of reranked result lists
        """
        reranked_all = []
        
        for query, results in query_result_pairs:
            reranked = self.rerank(query, results, top_k)
            reranked_all.append(reranked)
        
        return reranked_all


class HybridSearchEngine:
    """Hybrid search engine combining BM25 lexical search with vector similarity."""

    def __init__(
        self,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
        index_dir: Optional[str] = None,
        use_bm25: bool = True
    ):
        """
        Initialize hybrid search engine.

        Args:
            semantic_weight: Weight for semantic (vector) search results
            lexical_weight: Weight for lexical (BM25) search results  
            index_dir: Directory for Whoosh index (if None, uses in-memory)
            use_bm25: Whether to use rank-bm25 library instead of Whoosh
        """
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.use_bm25 = use_bm25
        
        # Document storage for search
        self.documents = []
        self.document_ids = []
        
        # BM25 components
        if use_bm25:
            self._init_bm25()
        else:
            self._init_whoosh(index_dir)
        
        logger.info(f"Initialized hybrid search with weights: semantic={semantic_weight:.2f}, lexical={lexical_weight:.2f}")

    def _init_bm25(self):
        """Initialize rank-bm25 for lexical search."""
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = None  # Will be initialized when documents are indexed
            self.tokenized_docs = []
            logger.info("Using rank-bm25 for lexical search")
        except ImportError as e:
            logger.error("rank-bm25 not available, falling back to Whoosh")
            self.use_bm25 = False
            self._init_whoosh(None)

    def _init_whoosh(self, index_dir: Optional[str]):
        """Initialize Whoosh for lexical search."""
        try:
            from whoosh.index import create_in, open_dir, exists_in
            from whoosh.fields import Schema, TEXT, ID, STORED
            from whoosh.qparser import QueryParser
            from whoosh.scoring import TF_IDF
            import tempfile
            import os

            # Define schema
            self.schema = Schema(
                id=ID(stored=True),
                content=TEXT(stored=True),
                url=STORED()
            )
            
            # Create or open index
            if index_dir is None:
                self.index_dir = tempfile.mkdtemp()
            else:
                self.index_dir = index_dir
                os.makedirs(index_dir, exist_ok=True)
            
            if exists_in(self.index_dir):
                from whoosh.index import open_dir
                self.index = open_dir(self.index_dir)
            else:
                self.index = create_in(self.index_dir, self.schema)
            
            self.query_parser = QueryParser("content", self.index.schema)
            logger.info(f"Whoosh index initialized at {self.index_dir}")
            
        except ImportError as e:
            logger.error(f"Whoosh not available: {e}")
            raise ImportError("Either rank-bm25 or Whoosh is required for hybrid search") from e

    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        import re
        # Simple tokenization: lowercase, remove punctuation, split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [token for token in text.split() if token.strip()]

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Index documents for both semantic and lexical search.
        
        Args:
            documents: List of documents with 'content', 'id', and optional 'url'
        """
        self.documents = documents
        self.document_ids = [doc.get('id', f'doc_{i}') for i, doc in enumerate(documents)]
        
        if self.use_bm25:
            self._index_documents_bm25(documents)
        else:
            self._index_documents_whoosh(documents)
        
        logger.info(f"Indexed {len(documents)} documents for hybrid search")

    def _index_documents_bm25(self, documents: List[Dict[str, Any]]):
        """Index documents using rank-bm25."""
        from rank_bm25 import BM25Okapi
        
        # Tokenize all documents
        self.tokenized_docs = []
        for doc in documents:
            content = doc.get('content', '')
            tokens = self.tokenize(content)
            self.tokenized_docs.append(tokens)
        
        # Create BM25 index
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
        else:
            self.bm25 = None
            logger.warning("No documents to index for BM25")

    def _index_documents_whoosh(self, documents: List[Dict[str, Any]]):
        """Index documents using Whoosh."""
        writer = self.index.writer()
        
        try:
            for i, doc in enumerate(documents):
                doc_id = doc.get('id', f'doc_{i}')
                content = doc.get('content', '')
                url = doc.get('url', '')
                
                writer.add_document(
                    id=doc_id,
                    content=content,
                    url=url
                )
            
            writer.commit()
            logger.info(f"Committed {len(documents)} documents to Whoosh index")
            
        except Exception as e:
            writer.cancel()
            logger.error(f"Failed to index documents in Whoosh: {e}")
            raise

    def lexical_search(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Perform lexical search using BM25 or Whoosh.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of (document_id, score) tuples
        """
        if self.use_bm25:
            return self._lexical_search_bm25(query, limit)
        else:
            return self._lexical_search_whoosh(query, limit)

    def _lexical_search_bm25(self, query: str, limit: int) -> List[Tuple[str, float]]:
        """Perform lexical search using rank-bm25."""
        if self.bm25 is None:
            logger.warning("BM25 index not initialized")
            return []
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Create (doc_id, score) pairs and sort
        doc_scores = [
            (self.document_ids[i], float(score))
            for i, score in enumerate(scores)
        ]
        
        # Sort by score descending and limit
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:limit]

    def _lexical_search_whoosh(self, query: str, limit: int) -> List[Tuple[str, float]]:
        """Perform lexical search using Whoosh."""
        try:
            with self.index.searcher() as searcher:
                # Parse and execute query
                parsed_query = self.query_parser.parse(query)
                results = searcher.search(parsed_query, limit=limit)
                
                # Extract document IDs and scores
                doc_scores = [
                    (result['id'], float(result.score))
                    for result in results
                ]
                
                return doc_scores
                
        except Exception as e:
            logger.error(f"Whoosh search failed: {e}")
            return []

    def fuse_results(
        self,
        semantic_results: List[SearchResult],
        lexical_results: List[Tuple[str, float]],
        limit: int
    ) -> List[SearchResult]:
        """
        Fuse semantic and lexical results using Reciprocal Rank Fusion.
        
        Args:
            semantic_results: Results from vector search
            lexical_results: Results from lexical search as (doc_id, score) pairs
            limit: Maximum number of final results
            
        Returns:
            Fused and ranked search results
        """
        # Create score dictionaries
        semantic_scores = {}
        lexical_scores = {}
        
        # Process semantic results (already ranked by relevance)
        for rank, result in enumerate(semantic_results):
            doc_id = result.document_id
            # Reciprocal rank fusion: score = weight / (rank + 1)
            semantic_scores[doc_id] = self.semantic_weight / (rank + 1)
        
        # Process lexical results (already scored by BM25/TF-IDF)
        # Normalize lexical scores to [0, 1] range
        if lexical_results:
            max_lexical_score = max(score for _, score in lexical_results)
            if max_lexical_score > 0:
                for rank, (doc_id, score) in enumerate(lexical_results):
                    normalized_score = score / max_lexical_score
                    lexical_scores[doc_id] = self.lexical_weight * normalized_score
        
        # Combine scores
        combined_scores = defaultdict(float)
        all_doc_ids = set(semantic_scores.keys()) | set(lexical_scores.keys())
        
        for doc_id in all_doc_ids:
            combined_scores[doc_id] = (
                semantic_scores.get(doc_id, 0.0) + 
                lexical_scores.get(doc_id, 0.0)
            )
        
        # Sort by combined score
        sorted_doc_ids = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        # Build final results, preserving original SearchResult objects where possible
        final_results = []
        semantic_results_dict = {r.document_id: r for r in semantic_results}
        
        for doc_id, final_score in sorted_doc_ids:
            if doc_id in semantic_results_dict:
                # Use existing SearchResult but update score
                original_result = semantic_results_dict[doc_id]
                fused_result = SearchResult(
                    content=original_result.content,
                    score=float(final_score),
                    metadata={
                        **original_result.metadata,
                        "semantic_score": semantic_scores.get(doc_id, 0.0),
                        "lexical_score": lexical_scores.get(doc_id, 0.0),
                        "fusion_method": "reciprocal_rank_fusion"
                    },
                    document_id=original_result.document_id,
                    source_url=original_result.source_url,
                )
                final_results.append(fused_result)
            else:
                # Create new SearchResult for lexical-only results
                # Find document content
                doc_content = ""
                doc_url = None
                for doc in self.documents:
                    if doc.get('id') == doc_id:
                        doc_content = doc.get('content', '')
                        doc_url = doc.get('url')
                        break
                
                fused_result = SearchResult(
                    content=doc_content,
                    score=float(final_score),
                    metadata={
                        "semantic_score": 0.0,
                        "lexical_score": lexical_scores.get(doc_id, 0.0),
                        "fusion_method": "reciprocal_rank_fusion"
                    },
                    document_id=doc_id,
                    source_url=doc_url,
                )
                final_results.append(fused_result)
        
        return final_results

    def hybrid_search(
        self,
        query: str,
        semantic_search_func: callable,
        limit: int = 10,
        semantic_limit: int = None,
        lexical_limit: int = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and lexical results.
        
        Args:
            query: Search query
            semantic_search_func: Function to perform semantic search
            limit: Final number of results to return
            semantic_limit: Number of semantic results to retrieve (default: limit * 2)
            lexical_limit: Number of lexical results to retrieve (default: limit * 2)
            
        Returns:
            Hybrid search results ranked by fused scores
        """
        if semantic_limit is None:
            semantic_limit = limit * 2
        if lexical_limit is None:
            lexical_limit = limit * 2
        
        try:
            # Perform semantic search
            semantic_results = semantic_search_func(query, semantic_limit)
            
            # Perform lexical search
            lexical_results = self.lexical_search(query, lexical_limit)
            
            # Fuse results
            fused_results = self.fuse_results(semantic_results, lexical_results, limit)
            
            logger.info(
                f"Hybrid search: {len(semantic_results)} semantic + {len(lexical_results)} lexical = {len(fused_results)} fused results"
            )
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to semantic search only
            return semantic_search_func(query, limit)


class SemanticSearch:
    """Semantic search over embeddings."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        model_type: Optional[str] = None,
        model_name: Optional[str] = None,
        vector_size: Optional[int] = None,
        enable_cross_encoder: bool = True,
        cross_encoder_model: Optional[str] = None,
        enable_hybrid_search: bool = True,
        semantic_weight: float = 0.7,
        lexical_weight: float = 0.3,
    ):
        """
        Initialize the semantic search.

        Args:
            collection_name: Name of the collection to search
            model_type: Type of embedding model
            model_name: Name of the embedding model to use
            vector_size: Vector dimension
            enable_cross_encoder: Whether to enable cross-encoder reranking
            cross_encoder_model: Specific cross-encoder model to use
        """
        config = get_config()
        self.collection_name = collection_name or config.qdrant.collection

        # Initialize embedding model
        self.embedding_model = get_embedding_model(model_type, model_name)

        # Initialize vector store
        self.vector_store = get_vector_store(self.collection_name, vector_size)

        # Initialize cross-encoder reranker
        self.cross_encoder = None
        self._enable_cross_encoder = enable_cross_encoder
        if enable_cross_encoder:
            try:
                model_to_use = cross_encoder_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self.cross_encoder = CrossEncoderReranker(model_to_use)
                logger.info("Cross-encoder reranker enabled")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder, disabling reranking: {e}")
                self._enable_cross_encoder = False

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
        use_cross_encoder: Optional[bool] = None,
        include_vectors: bool = False,
    ) -> List[SearchResult]:
        """
        Search for chunks similar to the query.

        Args:
            query: Query text
            limit: Maximum number of results
            threshold: Similarity threshold
            filter_conditions: Additional filter conditions
            rerank: Whether to rerank results using basic heuristics
            use_cross_encoder: Whether to use cross-encoder reranking (overrides rerank)
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

            # Determine if we should use cross-encoder or basic reranking
            should_use_cross_encoder = (
                use_cross_encoder if use_cross_encoder is not None 
                else (self._enable_cross_encoder and self.cross_encoder is not None)
            )
            
            # Get more results if we're doing any kind of reranking
            search_limit = limit * 3 if (rerank or should_use_cross_encoder) else limit

            # Search vector store
            raw_results = self.vector_store.search(
                query_vector=query_embedding,
                limit=search_limit,
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

            # Apply reranking based on available options
            if should_use_cross_encoder and len(search_results) > limit:
                # Use cross-encoder reranking (preferred)
                try:
                    search_results = self.cross_encoder.rerank(query, search_results, limit)
                    logger.info("Applied cross-encoder reranking")
                except Exception as e:
                    logger.warning(f"Cross-encoder reranking failed, falling back to basic reranking: {e}")
                    if rerank:
                        search_results = self._rerank_results(query, search_results, limit)
                    else:
                        search_results = search_results[:limit]
                        
            elif rerank and len(search_results) > limit:
                # Use basic heuristic reranking
                search_results = self._rerank_results(query, search_results, limit)
                logger.info("Applied heuristic reranking")
            else:
                # No reranking, just limit results
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
