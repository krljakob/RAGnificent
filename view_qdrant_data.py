"""
View data stored in Qdrant vector database.

Simple utility to view documents and metadata stored in a Qdrant collection.
First loads data into the in-memory database, then allows viewing and searching.
"""

import logging
from typing import Optional
import json
import argparse
from pprint import pprint

from RAGnificent.core.config import get_config
from RAGnificent.rag.vector_store import get_qdrant_client, QdrantConnectionError
from RAGnificent.rag.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(url: str = "https://www.rust-lang.org/learn", collection_name: str = "demo_collection"):
    """
    Load data into the in-memory Qdrant database

    Args:
        url: URL to process
        collection_name: Name of the collection to create
    """
    print(f"\nLoading data from URL: {url}")
    print("This step is required because you're using an in-memory database")
    print("Please wait while we process the data...\n")

    # Initialize the pipeline
    pipeline = Pipeline(collection_name=collection_name)

    # Run the complete pipeline
    result = pipeline.run_pipeline(
        url=url,
        limit=10,
        run_extract=True,
        run_chunk=True,
        run_embed=True,
        run_store=True,
    )

    # Display results
    if result["success"]:
        print(f"\nPipeline executed successfully")
        for key, count in result["document_counts"].items():
            print(f"- {key}: {count}")
    else:
        print("\nPipeline failed:")
        for step, status in result["steps"].items():
            print(f"- {step}: {status}")

    return result["success"]

def view_collection_info(collection_name: Optional[str] = None):
    """
    Display information about a specific collection

    Args:
        collection_name: Name of the collection (or None for default)
    """
    try:
        config = get_config()
        if collection_name is None:
            collection_name = config.qdrant.collection

        # Default to 'demo_collection' which was used in the simple_demo.py
        if collection_name is None:
            collection_name = "demo_collection"

        # Get client
        client = get_qdrant_client()

        # Get collection info
        collection_info = client.get_collection(collection_name=collection_name)
        print("\nCollection Information:")
        print(f"Name: {collection_name}")
        print(f"Vectors count: {collection_info.vectors_count}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        print(f"Distance: {collection_info.config.params.vectors.distance}")

        # Show schema if available
        if hasattr(collection_info, 'schema') and collection_info.schema:
            print("\nPayload Schema:")
            pprint(collection_info.schema)

    except QdrantConnectionError as e:
        logger.error(f"Connection error: {e}")
    except Exception as e:
        logger.error(f"Error viewing collection info: {e}")

def list_points(collection_name: Optional[str] = None, limit: int = 10):
    """
    List points/documents stored in a collection

    Args:
        collection_name: Name of the collection (or None for default)
        limit: Maximum number of points to retrieve
    """
    try:
        config = get_config()
        if collection_name is None:
            collection_name = config.qdrant.collection

        # Default to 'demo_collection' which was used in the simple_demo.py
        if collection_name is None:
            collection_name = "demo_collection"

        # Get client
        client = get_qdrant_client()

        # Scroll through points
        points = client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False  # Set to True if you want to see the actual vectors
        )[0]

        print(f"\nShowing up to {limit} points from collection '{collection_name}':")

        for i, point in enumerate(points):
            print(f"\n----- Document {i+1} -----")
            print(f"ID: {point.id}")

            # Print metadata and content
            if hasattr(point, 'payload') and point.payload:
                if 'metadata' in point.payload:
                    print("\nMetadata:")
                    pprint(point.payload['metadata'])

                if 'content' in point.payload:
                    print("\nContent:")
                    content = point.payload['content']
                    # Truncate long content for display
                    if len(content) > 300:
                        print(f"{content[:300]}...\n[Content truncated, total length: {len(content)} chars]")
                    else:
                        print(content)

                if other_fields := [
                    k
                    for k in point.payload.keys()
                    if k not in ('metadata', 'content')
                ]:
                    print("\nOther Fields:")
                    for field in other_fields:
                        print(f"{field}: {point.payload[field]}")

            print("-" * 50)

        return points

    except QdrantConnectionError as e:
        logger.error(f"Connection error: {e}")
    except Exception as e:
        logger.error(f"Error listing points: {e}")
        return []

def search_similar(query: str, collection_name: Optional[str] = None, limit: int = 5):
    """
    Search for similar documents based on a query

    Args:
        query: The search query text
        collection_name: Name of the collection (or None for default)
        limit: Maximum number of results to return
    """
    try:
        from RAGnificent.rag.embedding import embed_text, get_embedding_model

        config = get_config()
        if collection_name is None:
            collection_name = config.qdrant.collection

        # Default to 'demo_collection' which was used in the simple_demo.py
        if collection_name is None:
            collection_name = "demo_collection"

        # Get client
        client = get_qdrant_client()

        # Get embedding model
        embedding_model = get_embedding_model()

        # Embed the query
        query_embedding = embed_text(query, model=embedding_model)

        # Search for similar documents
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )

        print(f"\nTop {limit} matches for query: '{query}'")

        for i, result in enumerate(search_results):
            print(f"\n----- Match {i+1} (Score: {result.score:.4f}) -----")

            # Print metadata and content
            if hasattr(result, 'payload') and result.payload:
                if 'metadata' in result.payload:
                    print("\nMetadata:")
                    pprint(result.payload['metadata'])

                if 'content' in result.payload:
                    print("\nContent:")
                    content = result.payload['content']
                    # Truncate long content for display
                    if len(content) > 300:
                        print(f"{content[:300]}...\n[Content truncated, total length: {len(content)} chars]")
                    else:
                        print(content)

            print("-" * 50)

        return search_results

    except QdrantConnectionError as e:
        logger.error(f"Connection error: {e}")
    except Exception as e:
        logger.error(f"Error searching: {e}")
        return []

def interactive_search(collection_name: Optional[str] = None):
    """
    Interactive search interface

    Args:
        collection_name: Name of the collection (or None for default)
    """
    print("\n" + "=" * 30)
    print("Interactive Search Mode")
    print("Type 'exit' or 'quit' to return to main menu")
    print("=" * 30)

    while True:
        query = input("\nEnter search query: ")
        if query.lower() in ["exit", "quit"]:
            break

        search_similar(query, collection_name)

def export_collection(collection_name: Optional[str] = None, output_file: str = "qdrant_export.json"):
    """
    Export all documents from a collection to a JSON file

    Args:
        collection_name: Name of the collection (or None for default)
        output_file: Path to output JSON file
    """
    try:
        config = get_config()
        if collection_name is None:
            collection_name = config.qdrant.collection

        # Default to 'demo_collection' which was used in the simple_demo.py
        if collection_name is None:
            collection_name = "demo_collection"

        # Get client
        client = get_qdrant_client()

        # Get collection info
        collection_info = client.get_collection(collection_name=collection_name)
        total_points = collection_info.vectors_count

        # Scroll through all points
        points = client.scroll(
            collection_name=collection_name,
            limit=total_points,
            with_payload=True,
            with_vectors=False  # Set to True if you want to include vectors
        )[0]

        # Convert to serializable format
        export_data = []
        for point in points:
            point_data = {
                "id": point.id,
                "payload": point.payload
            }
            export_data.append(point_data)

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"\nExported {len(export_data)} documents to {output_file}")

    except QdrantConnectionError as e:
        logger.error(f"Connection error: {e}")
    except Exception as e:
        logger.error(f"Error exporting collection: {e}")

def main():
    """Main entry point with interactive menu."""
    print("=" * 80)
    print("Qdrant Data Viewer")
    print("=" * 80)

    # Default collection name
    collection_name = "demo_collection"

    # Always load data first because we're using in-memory database
    if not load_data(url="https://www.rust-lang.org/learn", collection_name=collection_name):
        print("Failed to load data. Exiting.")
        return

    # Show collection info
    view_collection_info(collection_name)

    # List documents
    list_points(collection_name, limit=5)

    # Interactive menu
    while True:
        print("\n" + "=" * 30)
        print("Options:")
        print("1. List more documents")
        print("2. Interactive search")
        print("3. Export collection to JSON")
        print("4. Exit")
        print("=" * 30)

        choice = input("\nEnter choice (1-4): ")

        if choice == "1":
            try:
                limit = int(input("Number of documents to list: "))
                list_points(collection_name, limit=limit)
            except ValueError:
                print("Please enter a valid number.")
        elif choice == "2":
            interactive_search(collection_name)
        elif choice == "3":
            output_file = input("Output file name [qdrant_export.json]: ") or "qdrant_export.json"
            export_collection(collection_name, output_file)
        elif choice == "4":
            print("Exiting. Remember, the in-memory database data will be lost!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
