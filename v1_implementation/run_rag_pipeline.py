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
import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
data_dir = Path(__file__).parent.parent / 'data'
os.makedirs(data_dir, exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the RAG pipeline")
    parser.add_argument(
        "--steps", 
        default="all", 
        help="Steps to run (comma-separated, e.g. '1,2,3,4,5') or 'all'"
    )
    parser.add_argument(
        "--url", 
        default="https://solana.com/docs", 
        help="Website to scrape for documentation"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=20, 
        help="Limit the number of documents to process"
    )
    return parser.parse_args()


def run_step_1(args):
    """Run document extraction."""
    logger.info("Step 1: Extracting documents...")
    from extraction import extract_documents
    
    documents = extract_documents(args.url, limit=args.limit)
    logger.info(f"Extracted {len(documents)} documents")
    return True


def run_step_2(args):
    """Run document chunking."""
    logger.info("Step 2: Chunking documents...")
    
    # Check if documents exist
    raw_docs_path = data_dir / 'raw_documents.json'
    if not raw_docs_path.exists():
        logger.error(f"Raw documents not found at {raw_docs_path}")
        return False
    
    from chunking import chunk_documents
    import json
    
    with open(raw_docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")
    return True


def run_step_3(args):
    """Run embedding generation."""
    logger.info("Step 3: Generating embeddings...")
    
    # Check if chunks exist
    chunks_path = data_dir / 'document_chunks.json'
    if not chunks_path.exists():
        logger.error(f"Document chunks not found at {chunks_path}")
        return False
    
    from embedding import embed_chunks
    import json
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    embedded_chunks = embed_chunks(chunks)
    logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
    return True


def run_step_4(args):
    """Run search demo."""
    logger.info("Step 4: Testing semantic search...")
    
    # Check if embedded chunks exist
    embedded_chunks_path = data_dir / 'embedded_chunks.json'
    if not embedded_chunks_path.exists():
        logger.error(f"Embedded chunks not found at {embedded_chunks_path}")
        return False
    
    from search import search_chunks, format_search_results
    import json
    
    with open(embedded_chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Run a test search
    test_query = "What is Solana?"
    results = search_chunks(test_query, chunks, top_k=3)
    
    print("\nSample Search Results:")
    print("=" * 50)
    print(f"Query: {test_query}")
    print(format_search_results(results))
    
    print("Interactive search is available by running 'python 4_search.py'")
    return True


def run_step_5(args):
    """Run chat interface."""
    logger.info("Step 5: Starting chat interface...")
    
    # Check if embedded chunks exist
    embedded_chunks_path = data_dir / 'embedded_chunks.json'
    if not embedded_chunks_path.exists():
        logger.error(f"Embedded chunks not found at {embedded_chunks_path}")
        return False
    
    from chat import chat
    import json
    
    with open(embedded_chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Verify chunks have embeddings
    chunks_with_embeddings = [chunk for chunk in chunks if 'embedding' in chunk]
    
    if not chunks_with_embeddings:
        logger.error("No chunks have embeddings")
        return False
    
    print("\nRAG-Powered Chat Interface")
    print("=" * 50)
    print("Ask questions about the documents (or 'quit' to exit)")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nNote: No OpenAI API key found. Using fallback response generation.")
    
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        if not query:
            continue
        
        # Get RAG-enhanced response
        response = chat(query, chunks_with_embeddings)
        
        # Display response
        print("\nAnswer:")
        print("-" * 50)
        print(response)
        print("-" * 50)
    
    return True


def main():
    """Run the RAG pipeline."""
    args = parse_args()
    
    # Determine which steps to run
    if args.steps.lower() == "all":
        steps = [1, 2, 3, 4, 5]
    else:
        steps = [int(s.strip()) for s in args.steps.split(",") if s.strip().isdigit()]
    
    # Update Python path for imports
    sys.path.append(str(Path(__file__).parent))
    
    # Run each step in sequence
    step_functions = {
        1: run_step_1,
        2: run_step_2,
        3: run_step_3,
        4: run_step_4,
        5: run_step_5
    }
    
    for step in steps:
        if step in step_functions:
            print(f"\n{'='*20} Running Step {step} {'='*20}\n")
            success = step_functions[step](args)
            if not success:
                logger.error(f"Step {step} failed, stopping pipeline")
                break
        else:
            logger.warning(f"Invalid step {step}, skipping")
    
    print("\nRAG pipeline completed!")


if __name__ == "__main__":
    main()
