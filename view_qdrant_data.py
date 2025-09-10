#!/usr/bin/env python3
"""
Utility script to view and inspect data stored in Qdrant vector database.

This script helps visualize what's in your RAGnificent vector collections.
"""

import contextlib
import json
import logging
from typing import Dict, List, Optional

from RAGnificent.core.config import get_config
from RAGnificent.rag.vector_store import get_vector_store

# configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_metadata(metadata: Dict) -> str:
    """Format metadata for display."""
    if not metadata:
        return "No metadata"

    lines = [
        f"  {key}: {value}"
        for key, value in metadata.items()
        if key in ["title", "document_url", "chunk_type", "chunk_index"]
    ]
    return "\n".join(lines) if lines else "No relevant metadata"


def view_collection_info(collection_name: Optional[str] = None):
    """View basic information about a collection."""
    config = get_config()
    collection = collection_name or config.qdrant.collection

    try:
        # get vector store
        vector_store = get_vector_store(collection)

        # get collection info
        count = vector_store.count_documents()

        if count == 0:
            return

        # get sample documents

        # use search with a broad query to get sample results
        from RAGnificent.rag.search import get_search

        search = get_search(collection)

        # get some sample results by searching for common terms
        sample_queries = ["the", "a", "and", "is", "of"]
        sample_results = []

        for query in sample_queries:
            try:
                results = search.search(query, limit=2, threshold=0.0)
                sample_results.extend(results)
                if len(sample_results) >= 5:
                    break
            except Exception as e:
                logger.debug(f"Query '{query}' failed: {e}")
                continue

        # remove duplicates by ID
        seen_ids = set()
        unique_results = []
        for result in sample_results:
            if hasattr(result, "id") and result.id not in seen_ids:
                seen_ids.add(result.id)
                unique_results.append(result)
            elif (
                hasattr(result, "metadata")
                and result.metadata.get("id") not in seen_ids
            ):
                seen_ids.add(result.metadata.get("id"))
                unique_results.append(result)

        # display sample documents
        for i, result in enumerate(unique_results[:5], 1):
            content = result.content if hasattr(result, "content") else str(result)
            content_preview = f"{content[:200]}..." if len(content) > 200 else content

            metadata = result.metadata if hasattr(result, "metadata") else {}

    except Exception as e:
        logger.error(f"Error viewing collection: {e}")


def search_collection(
    query: str, collection_name: Optional[str] = None, limit: int = 5
):
    """Search the collection with a query."""
    config = get_config()
    collection = collection_name or config.qdrant.collection

    try:
        from RAGnificent.rag.search import get_search

        search = get_search(collection)

        results = search.search(query, limit=limit, threshold=0.5)

        if not results:
            return

        for i, result in enumerate(results, 1):
            score = getattr(result, "score", 0.0)
            content = result.content if hasattr(result, "content") else str(result)
            content_preview = f"{content[:300]}..." if len(content) > 300 else content

            metadata = result.metadata if hasattr(result, "metadata") else {}
            source_url = metadata.get("document_url", "Unknown source")

    except Exception as e:
        logger.error(f"Error searching collection: {e}")


def list_collections():
    """List all available collections."""

    try:
        config = get_config()
        vector_store = get_vector_store()

        # this is a basic implementation - Qdrant client may have methods to list collections

        # try to get info about the default collection
        with contextlib.suppress(Exception):
            count = vector_store.count_documents()

    except Exception as e:
        logger.error(f"Error listing collections: {e}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="View and inspect RAGnificent Qdrant data"
    )
    parser.add_argument("--collection", "-c", help="Collection name (optional)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info command
    info_parser = subparsers.add_parser("info", help="View collection information")

    # search command
    search_parser = subparsers.add_parser("search", help="Search the collection")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--limit", "-l", type=int, default=5, help="Number of results"
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List collections")

    args = parser.parse_args()

    if args.command == "info" or args.command is None:
        view_collection_info(args.collection)
    elif args.command == "search":
        search_collection(args.query, args.collection, args.limit)
    elif args.command == "list":
        list_collections()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
