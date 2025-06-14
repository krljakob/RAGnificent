"""
RAGnificent Command Line Interface.

This module provides a modern, user-friendly CLI for RAGnificent using Typer.
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import typer
from rich.box import ROUNDED
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text


class KeyboardListener:
    """Listen for keyboard input in a separate thread."""
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.running = False

    def __enter__(self):
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.thread.join()

    def _listen(self):
        while self.running:
            try:
                char = sys.stdin.read(1)
                if char:
                    self.callback(char)
            except:
                pass
            time.sleep(0.1)

def keyboard_listener(callback: Callable[[str], None]):
    """Create a keyboard listener context manager."""
    return KeyboardListener(callback)
from typing_extensions import Annotated

from RAGnificent.core.logger import get_logger, setup_logger

# Initialize logger
logger = get_logger(__name__)

# Create Typer app
app = typer.Typer(
    name="rag",
    help="RAGnificent - A powerful RAG pipeline builder",
    add_completion=True,
    no_args_is_help=True,
)

# Create console instance
console = Console()

# Common options
verbose_option = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
debug_option = typer.Option(False, "--debug", "-d", help="Enable debug output")


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        from RAGnificent import __version__

        console.print(f"RAGnificent v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = verbose_option,
    debug: bool = debug_option,
):
    """RAGnificent - A powerful RAG pipeline builder."""
    # Configure logging based on verbosity
    log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    setup_logger(level=log_level)


def parse_domain_rate_limits(rate_limit_str: str) -> Dict[str, float]:
    """Parse domain rate limit string into a dict."""
    limits: Dict[str, float] = {}
    if not rate_limit_str:
        return limits
    for pair in rate_limit_str.split(","):
        if "=" in pair:
            domain, val = pair.split("=", 1)
            try:
                limits[domain.strip()] = float(val.strip())
            except Exception:
                continue
    return limits


@app.command()
def scrape(
    url: Optional[str] = typer.Argument(None, help="URL to scrape (ignored if --links-file is provided)", show_default=False),
    output_dir: Path = typer.Option(
        "output", "-o", "--output-dir", help="Output directory"
    ),
    format: str = typer.Option(
        "markdown", "-f", "--format", help="Output format (markdown, json, xml)"
    ),
    debug: bool = debug_option,
    use_sitemap: bool = typer.Option(
        False, "--sitemap", "-s", help="Use sitemap for URL discovery"
    ),
    parallel: bool = typer.Option(
        False, "--parallel", "-p", help="Enable parallel processing"
    ),
    workers: int = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers (auto-tuned if not set)",
    ),
    use_async: bool = typer.Option(
        None,
        "--async/--sync",
        help="Use async scraper for better performance (default: auto)",
    ),
    save_chunks: bool = typer.Option(
        False, "--save-chunks", help="Save content chunks for RAG"
    ),
    chunk_dir: Optional[Path] = typer.Option(
        None, "--chunk-dir", help="Directory to save chunks (defaults to output-dir)"
    ),
    domain_rate_limit: str = typer.Option(
        "",
        "--domain-rate-limit",
        help="Per-domain rate limits, e.g. 'example.com=2.0,python.org=5.0'",
    ),
    links_file: Optional[Path] = typer.Option(
        None, "--links-file", "-l", help="Path to a file containing links to scrape (defaults to links.txt if found)",
    ),
):
    """Scrape a website, or a list of websites from a links file, and convert content to the specified format."""
    import asyncio

    # Auto-tune workers if not set
    if workers is None:
        try:
            workers = min(32, (os.cpu_count() or 1) + 4)
        except Exception:
            workers = 8

    domain_limits = parse_domain_rate_limits(domain_rate_limit)

    # Determine if we should use a links file
    links_file_path = None
    if links_file is not None:
        links_file_path = links_file
    elif Path("links.txt").exists():
        links_file_path = Path("links.txt")

    # If both url and links_file are missing, error
    if not url and not links_file_path and not use_sitemap:
        console.print("[red]Error: You must provide either a URL, --links-file, or --sitemap.[/red]")
        raise typer.Exit(1)

    # Decide async/sync: async by default for multi-URL jobs
    multi_url = use_sitemap or parallel or links_file_path is not None
    if use_async is None:
        use_async = multi_url

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=2),
        Layout(name="footer", size=3)
    )

    # Create progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
        TransferSpeedColumn()
    )

    # Create stats table
    stats_table = Table(
        title="Scraping Stats",
        box=ROUNDED,
        expand=True
    )
    stats_table.add_column("Metric")
    stats_table.add_column("Value")

    # Create header panel
    header_panel = Panel(
        Text("RAGnificent Scraper", justify="center", style="bold blue"),
        border_style="blue",
        box=ROUNDED
    )

    with Live(layout, refresh_per_second=10, screen=True) as live:
        # Update layout
        layout["header"].update(header_panel)
        layout["main"].update(progress)
        layout["footer"].update(stats_table)

        # Add main scraping task
        task = progress.add_task("[cyan]Scraping...", total=100)

        # Add keyboard handler
        def handle_key(key):
            if key == "q":
                console.print("\n[red]Scraping cancelled by user[/red]")
                raise typer.Exit(0)
            if key == " ":
                if progress.tasks[task].finished:
                    return
                if progress.tasks[task].paused:
                    progress.start_task(task)
                    console.print("\n[green]Resumed scraping[/green]")
                else:
                    progress.stop_task(task)
                    console.print("\n[yellow]Paused scraping[/yellow]")
            elif key == "h":
                help_panel = Panel(
                    """[bold]Keyboard Controls:[/bold]
[cyan]Space[/cyan] - Pause/Resume
[cyan]Q[/cyan] - Quit
[cyan]H[/cyan] - Show this help""",
                    title="Help",
                    border_style="blue",
                    box=ROUNDED
                )
                live.update(help_panel)
                time.sleep(3)
                live.update(layout)

        # Start keyboard listener
        with keyboard_listener(handle_key):
            # Update stats table
            stats_table.add_row("Total URLs", "0")
            stats_table.add_row("Completed", "0")
            stats_table.add_row("Failed", "0")
            stats_table.add_row("Speed", "0 pages/sec")
            stats_table.add_row("Status", "Running")

        try:
            if use_async:
                from RAGnificent.core.async_scraper import AsyncMarkdownScraper

                async def run_async_scrape():

                    scraper = AsyncMarkdownScraper(
                        max_workers=workers,
                        domain_specific_limits=domain_limits,
                    )
                    if not scraper.rust_available:
                        console.print(
                            "[yellow]Warning: Rust extension not available. Falling back to slower Python implementation.[/yellow]"
                        )

                    if links_file_path is not None:
                        # Read links from file
                        with open(links_file_path, "r", encoding="utf-8") as f:
                            links = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                        if not links:
                            console.print(f"[red]No valid links found in {links_file_path}.[/red]")
                            return []

                        return await scraper.scrape_multiple_urls(
                            links,
                            output_dir=str(output_dir),
                            output_format=format,
                            save_chunks=save_chunks,
                            chunk_dir=str(chunk_dir) if chunk_dir else None,
                        )

                    if use_sitemap:
                        if url is None:
                            console.print("[red]Error: --sitemap requires a URL argument.[/red]")
                            raise typer.Exit(1)
                        return await scraper.scrape_by_sitemap(
                            url,
                            output_dir=str(output_dir),
                            output_format=format,
                            save_chunks=save_chunks,
                            chunk_dir=str(chunk_dir) if chunk_dir else None,
                        )

                    if url is None:
                        console.print("[red]Error: No URL provided for scraping.[/red]")
                        raise typer.Exit(1)

                    await scraper._scrape_and_save(
                        url,
                        str(output_dir),
                        format,
                        save_chunks,
                        str(chunk_dir) if chunk_dir else None,
                    )
                    return [url]

                # Run the async scraper in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    scraped_urls = loop.run_until_complete(run_async_scrape())
                finally:
                    loop.close()
            else:
                from RAGnificent.core.scraper import MarkdownScraper

                scraper = MarkdownScraper(
                    max_workers=workers,
                    domain_specific_limits=domain_limits,
                )
                if not scraper.rust_available:
                    console.print(
                        "[yellow]Warning: Rust extension not available. Falling back to slower Python implementation.[/yellow]"
                    )
                if links_file_path is not None:
                    scraped_urls = scraper.scrape_by_links_file(
                        str(links_file_path),
                        output_dir=str(output_dir),
                        output_format=format,
                        parallel=True,
                        max_workers=workers,
                        save_chunks=save_chunks,
                        chunk_dir=str(chunk_dir) if chunk_dir else None,
                    )
                elif use_sitemap:
                    if url is None:
                        console.print("[red]Error: --sitemap requires a URL argument.[/red]")
                        raise typer.Exit(1)
                    scraped_urls = scraper.scrape_by_sitemap(
                        url,
                        output_dir=str(output_dir),
                        output_format=format,
                        parallel=True,
                        max_workers=workers,
                        save_chunks=save_chunks,
                        chunk_dir=str(chunk_dir) if chunk_dir else None,
                    )
                else:
                    if url is None:
                        console.print("[red]Error: No URL provided for scraping.[/red]")
                        raise typer.Exit(1)
                    html_content = scraper.scrape_website(url)
                    if scraper.rust_available:
                        format_map = {
                            "markdown": scraper.OutputFormat.MARKDOWN,
                            "json": scraper.OutputFormat.JSON,
                            "xml": scraper.OutputFormat.XML,
                        }
                        output = scraper.convert_html(
                            html_content,
                            url,
                            format_map.get(format, scraper.OutputFormat.MARKDOWN),
                        )
                    else:
                        output, _ = scraper._convert_content(html_content, url, format)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    extension = "md" if format == "markdown" else format
                    output_file = output_dir / f"scraped.{extension}"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(output)
                    if save_chunks and format == "markdown":
                        chunks = scraper.chunker.create_chunks_from_markdown(
                            output, source_url=url
                        )
                        chunks_output_dir = chunk_dir or output_dir
                        chunks_output_dir.mkdir(parents=True, exist_ok=True)
                        for i, chunk in enumerate(chunks):
                            chunk_file = chunks_output_dir / f"scraped_chunk_{i:03d}.md"
                            with open(chunk_file, "w", encoding="utf-8") as f:
                                f.write(chunk.content)
                    scraped_urls = [url]

            progress.update(task, description="[green]Scraping completed!")
            console.print(f"\n✅ Successfully scraped {len(scraped_urls)} pages")

        except Exception as e:
            logger.error(f"Scraping failed: {e}", exc_info=True)
            raise typer.Exit(1) from e


@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Input file to convert"),
    output_file: Optional[Path] = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file (defaults to input file with new extension)",
    ),
    from_format: Optional[str] = typer.Option(
        None, "--from", help="Input format (auto-detected if not specified)"
    ),
    to_format: str = typer.Option(
        "markdown", "--to", help="Output format (markdown, json, xml, html)"
    ),
    chunk: bool = typer.Option(False, "--chunk", help="Enable semantic chunking"),
    chunk_size: int = typer.Option(
        1000, "--chunk-size", help="Maximum chunk size in characters"
    ),
    chunk_overlap: int = typer.Option(
        200, "--chunk-overlap", help="Overlap between chunks"
    ),
):
    """Convert documents between different formats."""
    from RAGnificent.ragnificent_rs import OutputFormat, convert_html
    from RAGnificent.utils.chunk_utils import ContentChunker

    console.print(f"📄 Converting {input_file} to {to_format}...")

    try:
        # Read input file
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Auto-detect input format if not specified
        if not from_format:
            if input_file.suffix.lower() in [".html", ".htm"]:
                from_format = "html"
            elif input_file.suffix.lower() == ".json":
                from_format = "json"
            elif input_file.suffix.lower() == ".xml":
                from_format = "xml"
            else:
                from_format = "markdown"

        # Convert based on output format
        if from_format == "html":
            if to_format == "markdown":
                converted = convert_html(content, output_format=OutputFormat.MARKDOWN)
            elif to_format == "json":
                converted = convert_html(content, output_format=OutputFormat.JSON)
            elif to_format == "xml":
                converted = convert_html(content, output_format=OutputFormat.XML)
            else:
                converted = content
        else:
            # For non-HTML inputs, we'll need to handle conversion differently
            converted = content  # Placeholder for now
            console.print(
                f"[yellow]Warning: Conversion from {from_format} to {to_format} not yet implemented[/yellow]"
            )

        # Apply chunking if requested
        if chunk and to_format == "markdown":
            chunker = ContentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunker.create_chunks_from_markdown(
                converted, source_url=str(input_file)
            )
            # Save chunks to separate files
            output_dir = output_file.parent if output_file else input_file.parent
            for i, chunk_data in enumerate(chunks):
                chunk_file = output_dir / f"{input_file.stem}_chunk_{i:03d}.{to_format}"
                with open(chunk_file, "w", encoding="utf-8") as f:
                    f.write(chunk_data.content)
            console.print(f"✅ Created {len(chunks)} chunks in {output_dir}")
        else:
            # Save converted content
            if not output_file:
                output_file = input_file.with_suffix(f".{to_format}")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(converted)

            console.print(f"✅ Converted file saved to {output_file}")

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


CONFIG_FILE_ARG = typer.Argument(..., help="Pipeline configuration file (YAML)")
INPUT_DIR_OPT = typer.Option(None, "-i", "--input", help="Input directory (overrides config)")

@app.command()
def pipeline(
    config_file: Path = CONFIG_FILE_ARG,
    input_dir: Optional[Path] = INPUT_DIR_OPT,
    output_dir: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output directory (overrides config)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be executed without running"
    ),
    continue_on_error: bool = typer.Option(
        False, "--continue", help="Continue pipeline on errors"
    ),
):
    """Execute a RAG pipeline from a configuration file."""
    import yaml  # type: ignore

    from RAGnificent.rag.pipeline import Pipeline

    console.print(f"🔄 Loading pipeline from {config_file}...")

    try:
        # Load pipeline configuration
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        # Override directories if provided
        if input_dir:
            config["input_dir"] = str(input_dir)
        if output_dir:
            config["output_dir"] = str(output_dir)

        # Show pipeline summary
        console.print("\n[bold]Pipeline Configuration:[/bold]")
        console.print(f"  - Name: {config.get('name', 'Unnamed Pipeline')}")
        console.print(f"  - Version: {config.get('version', '1.0')}")
        console.print(f"  - Steps: {len(config.get('steps', []))}")

        if dry_run:
            console.print("\n[yellow]DRY RUN - Showing pipeline steps:[/yellow]")
            for i, step in enumerate(config.get("steps", [])):
                console.print(f"\n  Step {i+1}: {step.get('name', 'Unnamed step')}")
                console.print(f"    Type: {step.get('type')}")
                console.print(f"    Config: {step.get('config', {})}")
            raise typer.Exit(0)

        # Execute pipeline
        pipeline = Pipeline(config, continue_on_error=continue_on_error)

        with Progress() as progress:
            task = progress.add_task(
                "Executing pipeline...", total=len(config.get("steps", []))
            )

            for step_result in pipeline.execute():
                progress.update(
                    task,
                    advance=1,
                    description=f"[green]{step_result['step_name']}[/green]",
                )

                if step_result["status"] == "error" and not continue_on_error:
                    console.print(
                        f"\n[red]Error in step '{step_result['step_name']}': {step_result['error']}[/red]"
                    )
                    raise typer.Exit(1)

        console.print("\n✅ Pipeline executed successfully!")

    except FileNotFoundError as e:
        logger.error(f"Pipeline configuration file not found: {config_file}")
        raise typer.Exit(1) from e
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in pipeline configuration: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@app.command()
def chat(
    query: str = typer.Argument(..., help="Query to send to the RAG system"),
    collection_name: Optional[str] = typer.Option(
        None, "-c", "--collection", help="Name of the vector collection"
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "-e", "--embedding-model", help="Name of embedding model to use"
    ),
):
    """Chat with the RAG system."""
    from RAGnificent.rag.chat import ChatSession

    console.print(Panel.fit("RAGnificent Chat", border_style="blue"))

    try:
        session = ChatSession(collection_name=collection_name, embedding_model=embedding_model)
        response = session.chat(query)

        console.print("\n[bold blue]You:[/bold blue]", query)
        console.print("\n[bold green]Assistant:[/bold green]")
        console.print(response)

    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@app.command()
def embed(
    input_path: Path = typer.Argument(..., help="Input file or directory to process"),
    output_dir: Path = typer.Option(
        "embeddings", "-o", "--output-dir", help="Output directory for embeddings"
    ),
    model_name: str = typer.Option(
        "all-MiniLM-L6-v2",
        "-m",
        "--model",
        help="Name of the embedding model to use",
    ),
    batch_size: int = typer.Option(
        32, "-b", "--batch-size", help="Batch size for embedding"
    ),
):
    """Generate embeddings for text content."""
    from RAGnificent.core.config import EmbeddingModelType
    from RAGnificent.rag.embedding import EmbeddingService, embed_texts_batched

    console.print("🔍 Generating embeddings...")

    try:
        service = EmbeddingService(model_name=model_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            # Process single file
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()

            embedding = service.embed_chunk(content)

            import json
            with open(output_dir / f"{input_path.stem}_embeddings.json", "w", encoding="utf-8") as f:
                json.dump(embedding, f)
        else:
            # Process directory
            import glob
            import json

            files = list(input_path.glob("**/*.md")) + list(input_path.glob("**/*.txt"))

            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()

                embedding = service.embed_chunk(content)

                relative_path = file.relative_to(input_path)
                output_file = output_dir / f"{relative_path.stem}_embeddings.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(embedding, f)

        console.print(f"✅ Embeddings saved to {output_dir}")

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "-p", "--port", help="Port to bind to"),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development"
    ),
):
    """Start the RAGnificent API server."""
    import uvicorn

    console.print("🚀 Starting RAGnificent API server...")
    console.print(f"   - Host: {host}")
    console.print(f"   - Port: {port}")
    console.print(f"   - Reload: {reload}")
    console.print("\n📚 API Documentation:")
    console.print(f"   - http://{host}:{port}/docs")
    console.print(f"   - http://{host}:{port}/redoc")

    uvicorn.run(
        "RAGnificent.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    app()
