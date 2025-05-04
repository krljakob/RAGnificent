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
import json
import logging
import os
import sys
import time
from pathlib import Path

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "search", "chat"],
        default="pipeline",
        help="Operation mode (pipeline, search, chat)"
    )
    parser.add_argument(
        "--url",
        default="https://www.rust-lang.org/learn",
        help="URL to process"
    )
    parser.add_argument(
        "--collection",
        default="demo_collection",
        help="Vector database collection name"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results or documents to process"
    )
    
    return parser.parse_args()

def run_pipeline_mode(args):
    """Run the complete RAG pipeline."""
    print("\n===== Running RAG Pipeline =====")
    
    # Initialize the pipeline
    pipeline = Pipeline(
        collection_name=args.collection
    )
    
    print(f"Processing URL: {args.url}")
    print("This may take a moment...\n")
    
    # Run the complete pipeline
    result = pipeline.run_pipeline(
        url=args.url,
        limit=args.limit,
        run_extract=True,
        run_chunk=True,
        run_embed=True,
        run_store=True
    )
    
    # Display results
    if result["success"]:
        print("\n✅ Pipeline completed successfully!")
        print("\nDocument counts:")
        for key, count in result["document_counts"].items():
            print(f"  - {key}: {count}")
    else:
        print("\n❌ Pipeline failed!")
        print("\nStep status:")
        for step, status in result["steps"].items():
            status_str = "✅ Success" if status else "❌ Failed"
            print(f"  - {step}: {status_str}")

def run_search_mode(args):
    """Run the search interface."""
    print("\n===== RAGnificent Search Mode =====")
    
    # Initialize the pipeline
    pipeline = Pipeline(
        collection_name=args.collection
    )
    
    while True:
        # Get search query
        print("\nEnter search query (or 'exit' to quit):")
        query = input("> ")
        
        if query.lower() in ["exit", "quit"]:
            break
            
        # Process the query
        print("\nSearching knowledge base...")
        results = pipeline.search_documents(
            query=query, 
            limit=args.limit,
            threshold=0.6
        )
        
        # Display results
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results, 1):
                source_url = result.get("source_url", "Unknown source")
                score = result.get("score", 0)
                
                # Get short snippet
                content = result.get("content", "")
                snippet = content[:200] + "..." if len(content) > 200 else content
                
                print(f"\n{i}. Score: {score:.2f}")
                print(f"   Source: {source_url}")
                print(f"   Snippet: {snippet}")
        else:
            print("\nNo results found")

def run_chat_mode(args):
    """Run the chat interface."""
    print("\n===== RAGnificent Chat Mode =====")
    
    # Check if OpenAI is installed
    try:
        import openai
    except ImportError:
        print("\nError: OpenAI package not installed.")
        print("Install it with: pip install openai")
        print("You'll also need to set your OPENAI_API_KEY environment variable.")
        return
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY environment variable not set.")
        print("You can still continue, but the chat functionality may not work correctly.")
    
    # Initialize the pipeline
    pipeline = Pipeline(
        collection_name=args.collection
    )
    
    while True:
        # Get query
        print("\nEnter your question (or 'exit' to quit):")
        query = input("> ")
        
        if query.lower() in ["exit", "quit"]:
            break
            
        # Process the query
        print("\nGenerating response...")
        result = pipeline.query_with_context(
            query=query,
            limit=args.limit,
            threshold=0.6
        )
        
        # Display response
        print("\n=== Response ===")
        print(result["response"])
        
        # Show sources if available
        if result.get("context"):
            print("\n=== Sources ===")
            for i, ctx in enumerate(result["context"], 1):
                url = ctx.get("source_url", "Unknown source")
                print(f"{i}. {url}")

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
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("An error occurred")
        sys.exit(1)
