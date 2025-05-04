#!/usr/bin/env python
"""
RAGnificent Simple Demo

This script demonstrates the core functionality of the consolidated RAGnificent implementation.
It doesn't require any external dependencies beyond what's needed for the core functionality.

Usage:
    python simple_demo.py [--mode MODE] [--url URL]

Options:
    --mode MODE      Operation mode (pipeline, search, chat) [default: pipeline]
    --url URL        URL to process [default: https://www.rust-lang.org/learn]
"""

import argparse
import logging
import os
import sys

from RAGnificent.core.config import get_config
from RAGnificent.rag.pipeline import Pipeline

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("ragnificent.demo")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAGnificent Simple Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "search", "chat"],
        default="pipeline",
        help="Operation mode (pipeline, search, chat)",
    )
    parser.add_argument(
        "--url", default="https://www.rust-lang.org/learn", help="URL to process"
    )
    parser.add_argument(
        "--collection",
        default="demo_collection",
        help="Vector database collection name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results or documents to process",
    )

    return parser.parse_args()


def run_pipeline_mode(args):
    """Run the complete RAG pipeline."""

    # Initialize the pipeline
    pipeline = Pipeline(collection_name=args.collection)


    # Run the complete pipeline
    result = pipeline.run_pipeline(
        url=args.url,
        limit=args.limit,
        run_extract=True,
        run_chunk=True,
        run_embed=True,
        run_store=True,
    )

    # Display results
    if result["success"]:
        for _key, _count in result["document_counts"].items():
            pass
    else:
        for _step, _status in result["steps"].items():
            pass


def run_search_mode(args):
    """Run the search interface."""

    # Initialize the pipeline
    pipeline = Pipeline(collection_name=args.collection)

    while True:
        # Get search query
        query = input("> ")

        if query.lower() in ["exit", "quit"]:
            break

        # Process the query
        if results := pipeline.search_documents(
            query=query, limit=args.limit, threshold=0.6
        ):
            for _i, result in enumerate(results, 1):
                result.get("source_url", "Unknown source")
                result.get("score", 0)

                # Get short snippet
                content = result.get("content", "")
                f"{content[:200]}..." if len(content) > 200 else content

        else:
            pass


def run_chat_mode(args):
    """Run the chat interface."""

    # Check if OpenAI is installed
    try:
        import openai
    except ImportError:
        return

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        pass

    # Initialize the pipeline
    pipeline = Pipeline(collection_name=args.collection)

    while True:
        # Get query
        query = input("> ")

        if query.lower() in ["exit", "quit"]:
            break

        # Process the query
        result = pipeline.query_with_context(
            query=query, limit=args.limit, threshold=0.6
        )

        # Display response

        # Show sources if available
        if result.get("context"):
            for _i, ctx in enumerate(result["context"], 1):
                ctx.get("source_url", "Unknown source")


def main():
    """Main entry point."""
    args = parse_args()

    # Ensure data directory exists
    config = get_config()
    os.makedirs(config.data_dir, exist_ok=True)

    # Run selected mode
    if args.mode == "pipeline":
        run_pipeline_mode(args)
    elif args.mode == "search":
        run_search_mode(args)
    elif args.mode == "chat":
        run_chat_mode(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        logger.exception("An error occurred")
        sys.exit(1)
