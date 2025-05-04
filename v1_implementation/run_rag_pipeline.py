#!/usr/bin/env python
"""
RAG Pipeline Runner

This script runs the complete RAG pipeline in sequence:
1. Extract documents from a website
2. Chunk documents into manageable pieces
3. Generate embeddings for each chunk
4. Enable search across the document chunks
5. Provide a chat interface with RAG-enhanced responses

Usage:
    python run_rag_pipeline.py [--steps STEPS] [--url URL] [--limit LIMIT]

Options:
    --steps STEPS    Steps to run (comma-separated, e.g. "1,2,3,4,5") [default: all]
    --url URL        Website to scrape for documentation [default: https://solana.com/docs]
    --limit LIMIT    Limit the number of documents to process [default: 20]
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

# Import pipeline steps
from v1_implementation.extraction import extract_documents

# Initialize logger
logger = logging.getLogger(__name__)
console = Console()

def validate_url(url: str) -> str:
    """Validate that URL has http/https scheme and a netloc."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL format: {url}")
    return url


def parse_args():
    """Parse and validate command line arguments with enhanced error handling."""
    try:
        parser = argparse.ArgumentParser(
            description="RAG Pipeline Runner",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--steps",
                          default="all",
                          help="Steps to run (comma-separated, e.g. '1,2,3,4,5')")
        parser.add_argument("--url",
                          default="https://www.terminaltrove.com/",
                          help="Website to scrape for documentation",
                          type=validate_url)
        parser.add_argument("--limit",
                          type=int,
                          default=20,
                          help="Limit the number of documents to process",
                          choices=range(1, 101))
        parser.add_argument("--debug",
                          action="store_true",
                          help="Enable debug logging")
        parser.add_argument("--config",
                          default="config.yaml",
                          help="Path to configuration file")

        args = parser.parse_args()

        # Validate steps format
        if args.steps != "all":
            try:
                steps = [int(s) for s in args.steps.split(",")]
                if not all(1 <= s <= 5 for s in steps):
                    raise ValueError("Steps must be between 1-5")
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"Invalid steps format: {e}") from e

        return args
    except Exception as e:
        logger.error(f"Argument parsing failed: {str(e)}")
        raise

from pathlib import Path


def main():
    """Run the RAG pipeline."""
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.debug:
        logger.debug("Debug mode enabled - verbose logging activated")

    # Ensure data directory exists
    data_dir = Path(__file__).parent.parent / 'data'
    os.makedirs(data_dir, exist_ok=True)

    # Determine which steps to run
    steps_to_run = list(range(1, 6)) if args.steps == "all" else [int(s) for s in args.steps.split(",")]


def run_step_1(args):
    """Run document extraction."""
    logger.info("Step 1: Extracting documents...")

    try:
        documents = extract_documents(args.url, limit=args.limit, debug=args.debug)
        if not documents:
            logger.warning("No documents were extracted - check the source URL and sitemap")
            return False
        logger.info(f"Extracted {len(documents)} documents")
        return True
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Document extraction failed: {str(e)}\nTraceback:\n{error_trace}")
        return False


def run_step_2(args):
    """Run document chunking."""
    logger.info("Step 2: Chunking documents...")

    # Check if documents exist
    raw_docs_path = data_dir / 'raw_documents.json'
    if not raw_docs_path.exists():
        logger.error(f"Raw documents not found at {raw_docs_path}")
        return False

    import json

    from chunking import chunk_documents

    with open(raw_docs_path, 'r', encoding='utf-8') as f:
        try:
            documents = json.load(f)
        except json.JSONDecodeError:
            documents = []

    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    with open(data_dir / 'document_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return True


def run_step_3(args):
    """Run embedding generation and Qdrant storage."""
    logger.info("Step 3: Generating embeddings and storing in Qdrant...")

    # Check if chunks exist
    chunks_path = data_dir / 'document_chunks.json'
    if not chunks_path.exists():
        logger.error(f"Document chunks not found at {chunks_path}")
        return False

    import json

    from embedding import embed_chunks, get_qdrant_client

    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Initialize Qdrant client
    get_qdrant_client()

    # Embed chunks and store in Qdrant
    embedded_chunks = embed_chunks(chunks)

    # Verify embeddings were created
    chunks_with_embeddings = [chunk for chunk in embedded_chunks if 'embedding' in chunk]
    if not chunks_with_embeddings:
        logger.error("No chunks received embeddings")
        return False

    logger.info(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")

    # Save as fallback (optional)
    with open(data_dir / 'embedded_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

    return True


def run_step_4(args):
    """Run Qdrant semantic search demo."""
    logger.info("Step 4: Testing Qdrant semantic search...")

    from search import search_chunks

    # Run a test search
    test_query = "What is Solana?"
    results = search_chunks(test_query, top_k=3)

    if not results:
        logger.error("No results returned from Qdrant search")
        return False


    return True


def run_step_5(args):
    """Run Qdrant-powered chat interface."""
    logger.info("Step 5: Starting Qdrant-powered chat interface...")

    from chat import chat
    from search import search_chunks


    if not os.environ.get("OPENAI_API_KEY"):
        pass

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break

        if not query:
            continue

        if relevant_chunks := search_chunks(query, top_k=5):
            # Get RAG-enhanced response
            chat(query, [chunk for chunk, _ in relevant_chunks])

            # Display response

    return True


def main():
    """Run the RAG pipeline."""
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Ensure data directory exists
    data_dir = Path(__file__).parent.parent / 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Determine which steps to run
    steps_to_run = list(range(1, 6)) if args.steps == "all" else [int(s) for s in args.steps.split(",")]


    # Run each step in sequence
    step_functions = {
        1: run_step_1,
        # Uncomment these as they are implemented
        # 2: run_step_2,
        # 3: run_step_3,
        # 4: run_step_4,
        # 5: run_step_5
    }
    
    for step in steps_to_run:
        if step not in step_functions:
            logger.warning(f"Step {step} not implemented yet, skipping")
            continue
            
        console.print(
            f"\n==================== Running Step {step} ====================",
            style="bold green"
        )
        
        if not step_functions[step](args):
            logger.error(f"Step {step} failed, stopping pipeline")
            sys.exit(1)



if __name__ == "__main__":
    main()
