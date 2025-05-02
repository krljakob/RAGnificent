#!/usr/bin/env python
"""Semantic search module for RAG implementation using Qdrant

This script provides search functionality to find relevant document chunks
using Qdrant vector database.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from embedding import get_qdrant_client, search_qdrant

# Load environment variables for API keys
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import hashlib
from datetime import datetime
from functools import lru_cache

# Cache size for search results (1000 queries)
SEARCH_CACHE_SIZE = 1000

# Rate limiting configuration
MAX_SEARCHES_PER_MINUTE = 30
search_timestamps = []

def preprocess_query(query: str) -> str:
    """Normalize and clean search queries."""
    # Lowercase, strip whitespace, remove special chars
    query = query.lower().strip()
    query = ''.join(c for c in query if c.isalnum() or c.isspace())
    return ' '.join(query.split())  # Remove extra spaces

def check_rate_limit():
    """Enforce rate limiting for search requests."""
    global search_timestamps

    # Remove old timestamps (older than 1 minute)
    now = datetime.now()
    search_timestamps = [t for t in search_timestamps
                        if (now - t).total_seconds() < 60]

    if len(search_timestamps) >= MAX_SEARCHES_PER_MINUTE:
        raise RuntimeError("Search rate limit exceeded (30 searches/minute)")

    search_timestamps.append(now)

@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def get_query_cache_key(query: str, top_k: int) -> str:
    """Generate consistent cache key for queries."""
    return hashlib.md5(f"{query}-{top_k}".encode()).hexdigest()

def search_chunks(
    query: str,
    chunks: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 5,
    use_cache: bool = True
) -> List[Tuple[Dict[str, Any], float]]:
    """Search for chunks similar to query using Qdrant with enhanced features.

    Args:
        query: Search query
        chunks: Optional list of chunks (not used with Qdrant)
        top_k: Number of top results to return
        use_cache: Whether to use cached results

    Returns:
        List of (chunk_dict, similarity_score) tuples

    Raises:
        RuntimeError: If rate limit is exceeded
        ValueError: For invalid queries
        ConnectionError: For Qdrant connection issues
    """
    try:
        # Validate and preprocess query
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        processed_query = preprocess_query(query)
        logger.info(f"Processing search query: {processed_query}")

        # Check rate limiting
        check_rate_limit()

        # Generate cache key
        get_query_cache_key(processed_query, top_k) if use_cache else None

        # Load chunks and embeddings from file if chunks not provided
        if chunks is None:
            embeddings_file = Path(__file__).parent.parent / 'data' / 'embeddings.json'
            if not embeddings_file.exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

            with open(embeddings_file) as f:
                data = json.load(f)
                chunks = data.get('chunks', [])

        # Simple vector search implementation (fallback if Qdrant not available)
        if not chunks:
            logger.warning("No chunks available for search")
            return []

        # Try using Qdrant for search
        try:
            results = search_qdrant(processed_query, top_k=top_k)
            logger.info("Used Qdrant for search")
        except Exception as e:
            logger.warning(f"Qdrant search failed: {e}, falling back to local search")
            # Perform simple local search (word overlap)
            results = []
            for chunk in chunks:
                # Calculate a simple relevance score based on word overlap
                query_words = set(processed_query.lower().split())
                chunk_text = chunk.get('text', '').lower()
                word_matches = sum(bool(word in chunk_text)
                               for word in query_words)
                score = word_matches / max(1, len(query_words))
                if score > 0:
                    results.append((chunk, score))

            # Sort by score descending
            results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
            logger.info("Used fallback local search")

        # Format the results
        formatted_results = results
        logger.info(f"Found {len(formatted_results)} relevant chunks")
        return formatted_results

    except RuntimeError as e:
        logger.error(f"Rate limit exceeded: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid query: {e}")
        raise
    except ConnectionError as e:
        logger.error(f"Qdrant connection error: {e}")
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise


def format_search_results(results: List[Tuple[Dict[str, Any], float]]) -> str:
    """Format search results for display.

    Args:
        results: List of (chunk, similarity_score) tuples

    Returns:
        Formatted string of results
    """
    if not results:
        return "No relevant documents found."

    formatted = "Search Results:\n" + "=" * 50 + "\n\n"

    for i, (chunk, score) in enumerate(results, 1):
        # Get chunk information
        title = chunk.get('title', 'Untitled')
        url = chunk.get('url', '')
        text = chunk.get('text', '')

        # Format chunk
        formatted += f"Result {i} (Relevance: {score:.4f})\n"
        formatted += f"Title: {title}\n"
        if url:
            formatted += f"Source: {url}\n"
        formatted += "\n"

        # Show preview of text (first 300 chars)
        preview = f"{text[:300]}..." if len(text) > 300 else text
        formatted += f"{preview}\n\n"
        formatted += "-" * 50 + "\n\n"

    return formatted


if __name__ == "__main__":
    # Initialize Qdrant client
    client = get_qdrant_client()

    # Interactive search

    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break

        if not query:
            continue

        # Search Qdrant
        results = search_chunks(query)

        # Display results
