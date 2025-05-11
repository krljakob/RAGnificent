#!/usr/bin/env python
"""
RAGnificent Integrated Demo

This script demonstrates the integrated RAG pipeline that combines:
- Rust-optimized scraping and processing from the new RAGnificent implementation

# Use relative imports for internal modules
# Import fix applied
sys.path.insert(0, str(Path(__file__).parent.parent))
- Complete RAG pipeline functionality from the v1 implementation

Usage:
    python -m RAGnificent.rag.demo [--url URL] [--steps STEPS] [--limit LIMIT] [--mode MODE]

Options:
    --url URL        Website to scrape [default: https://www.rust-lang.org/learn]
    --steps STEPS    Steps to run (comma-separated, e.g. "1,2,3,4" or "all") [default: all]
    --limit LIMIT    Limit the number of documents to process [default: 10]
    --mode MODE      Pipeline mode (scrape, chat, search) [default: scrape]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from rag.chat import RAGChat
from rag.pipeline import RAGPipeline
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("ragnificent.demo")
console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAGnificent Integrated Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="https://www.rust-lang.org/learn",
        help="Website to scrape for documentation",
    )
    parser.add_argument(
        "--steps",
        default="all",
        help="Steps to run (comma-separated, e.g. '1,2,3,4' or 'all')",
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Limit the number of documents to process"
    )
    parser.add_argument(
        "--mode",
        choices=["scrape", "chat", "search"],
        default="scrape",
        help="Pipeline mode (scrape, chat, search)",
    )
    parser.add_argument(
        "--collection",
        default="ragnificent_demo",
        help="Vector database collection name",
    )

    return parser.parse_args()


def run_pipeline_mode(args):
    """Run the complete pipeline."""
    console.print("[bold green]Running Integrated RAG Pipeline[/bold green]")

    # Initialize the pipeline
    pipeline = RAGPipeline(collection_name=args.collection)

    # Determine which steps to run
    steps_to_run = {"extract": True, "chunk": True, "embed": True, "store": True}

    if args.steps != "all":
        _configure_pipeline_steps(steps_to_run, args)
    if success := pipeline.run_pipeline(
        url=args.url,
        limit=args.limit,
        run_extract=steps_to_run["extract"],
        run_chunk=steps_to_run["chunk"],
        run_embed=steps_to_run["embed"],
        run_store=steps_to_run["store"],
    ):
        console.print("[bold green]Pipeline completed successfully![/bold green]")
    else:
        console.print("[bold red]Pipeline failed![/bold red]")


def _configure_pipeline_steps(steps_to_run, args):
    # Reset all steps to False
    for step in steps_to_run:
        steps_to_run[step] = False

    # Enable only requested steps
    step_nums = [int(s) for s in args.steps.split(",")]
    if 1 in step_nums:
        steps_to_run["extract"] = True
    if 2 in step_nums:
        steps_to_run["chunk"] = True
    if 3 in step_nums:
        steps_to_run["embed"] = True
    if 4 in step_nums:
        steps_to_run["store"] = True


def run_chat_mode(args):
    """Run the chat interface."""
    console.print("[bold green]RAGnificent Chat Mode[/bold green]")
    console.print("[italic]Type 'exit' or 'quit' to end the session[/italic]")

    # Initialize the chat
    chat = RAGChat(collection_name=args.collection)

    while True:
        query = Prompt.ask("\n[bold blue]You")
        if query.lower() in ["exit", "quit"]:
            break

        # Process the query
        with console.status("[bold yellow]Searching knowledge base..."):
            result = chat.chat(query)

        # Display response
        console.print(f"\n[bold green]RAGnificent[/bold green]: {result['response']}")

        # Show sources if available
        if result["context"]:
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            for i, ctx in enumerate(result["context"], 1):
                url = ctx.get("url", "Unknown")
                score = ctx.get("score", 0)
                console.print(f"  {i}. [link={url}]{url}[/link] (Score: {score:.2f})")


def run_search_mode(args):
    """Run the search interface."""
    console.print("[bold green]RAGnificent Search Mode[/bold green]")
    console.print("[italic]Type 'exit' or 'quit' to end the session[/italic]")

    # Initialize the pipeline
    pipeline = RAGPipeline(collection_name=args.collection)

    while True:
        query = Prompt.ask("\n[bold blue]Search")
        if query.lower() in ["exit", "quit"]:
            break

        # Process the query
        with console.status("[bold yellow]Searching knowledge base..."):
            results = pipeline.search_documents(query, limit=5)

        # Display results
        if results:
            console.print(f"\n[bold green]Found {len(results)} results:[/bold green]")
            for i, result in enumerate(results, 1):
                payload = result.get("payload", {})
                content = payload.get("content", "")
                source_url = payload.get("source_url", "")
                score = result.get("score", 0)

                # Truncate content for display
                display_content = (
                    f"{content[:300]}..." if len(content) > 300 else content
                )

                console.print(
                    f"\n[bold cyan]{i}. Score: {score:.2f} - [link={source_url}]{source_url}[/link][/bold cyan]"
                )
                console.print(f"{display_content}")
        else:
            console.print("\n[bold yellow]No results found[/bold yellow]")


def main():
    """Main entry point."""
    args = parse_args()

    # Ensure data directory exists
    data_dir = Path.cwd() / "data"
    os.makedirs(data_dir, exist_ok=True)

    # Run selected mode
    if args.mode == "scrape":
        run_pipeline_mode(args)
    elif args.mode == "chat":
        run_chat_mode(args)
    elif args.mode == "search":
        run_search_mode(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Operation cancelled by user[/bold yellow]")
        sys.exit(0)
    except Exception:
        logger.exception("An error occurred")
        sys.exit(1)
