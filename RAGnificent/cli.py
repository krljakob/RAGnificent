"""
RAGnificent Command Line Interface.

This module provides a modern, user-friendly CLI for RAGnificent using Typer.
"""

import asyncio
from pathlib import Path
from typing import List, Optional

import typer
import uvicorn
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from RAGnificent import __version__
from RAGnificent.core.async_scraper import AsyncMarkdownScraper
from RAGnificent.core.logging import get_logger, setup_logger
from RAGnificent.core.scraper import MarkdownScraper
from RAGnificent.rag.chat import RAGChat
from RAGnificent.rag.embedding import EmbeddingService
from RAGnificent.rag.pipeline import Pipeline
from RAGnificent.ragnificent_rs import (
    OutputFormat,
    chunk_markdown,
    convert_html_to_format,
    convert_html_to_markdown,
)
from RAGnificent.utils.chunk_utils import ContentChunker

# Initialize logger
logger = get_logger(__name__)

# Create Typer app
app = typer.Typer(
    name="rag",
    help="RAGnificent CLI for web scraping and content processing",
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
    """Configure logging and global options."""
    # Configure logging based on verbosity
    log_level = "DEBUG" if debug else ("INFO" if verbose else "WARNING")
    setup_logger(level=log_level)


@app.command()
def scrape(
    url: str = typer.Argument(..., help="URL to scrape"),
    output_dir: Path = typer.Option(
        "output", "-o", "--output-dir", help="Output directory"
    ),
    format: str = typer.Option(
        "markdown", "-f", "--format", help="Output format (markdown, json, xml)"
    ),
    use_sitemap: bool = typer.Option(
        False, "--sitemap", "-s", help="Use sitemap for URL discovery"
    ),
    parallel: bool = typer.Option(
        False, "--parallel", "-p", help="Enable parallel processing"
    ),
    workers: int = typer.Option(
        4, "--workers", "-w", help="Number of parallel workers"
    ),
    use_async: bool = typer.Option(
        True, "--async/--sync", help="Use async scraper for better performance"
    ),
    save_chunks: bool = typer.Option(
        False, "--save-chunks", help="Save content chunks for RAG"
    ),
    chunk_dir: Optional[Path] = typer.Option(
        None, "--chunk-dir", help="Directory to save chunks (defaults to output-dir)"
    ),
):
    """Scrape a website and convert content to the specified format."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Scraping...", total=None)

        try:
            if use_async:
                scraped_urls = asyncio.run(
                    _async_scrape(
                        url,
                        output_dir,
                        format,
                        use_sitemap,
                        parallel,
                        workers,
                        save_chunks,
                        chunk_dir,
                        progress,
                        task,
                    )
                )
            else:
                scraped_urls = _sync_scrape(
                    url,
                    output_dir,
                    format,
                    use_sitemap,
                    parallel,
                    workers,
                    save_chunks,
                    chunk_dir,
                    progress,
                    task,
                )

            progress.update(task, description="[green]Scraping completed!")
            console.print(f"\n✅ Successfully scraped {len(scraped_urls)} pages")

        except Exception as e:
            logger.error(f"Scraping failed: {e}", exc_info=debug)
            raise typer.Exit(1) from e


async def _async_scrape(
    url,
    output_dir,
    format,
    use_sitemap,
    parallel,
    workers,
    save_chunks,
    chunk_dir,
    progress,
    task,
):
    """Async scraping implementation."""
    async with AsyncMarkdownScraper(max_workers=workers) as scraper:
        if use_sitemap:
            scraped_urls = await scraper.scrape_by_sitemap(
                url,
                output_dir=str(output_dir),
                output_format=format,
                save_chunks=save_chunks,
                chunk_dir=str(chunk_dir) if chunk_dir else None,
            )
        else:
            # Single URL mode
            await scraper._scrape_and_save(
                url,
                str(output_dir),
                format,
                save_chunks,
                str(chunk_dir) if chunk_dir else None,
            )
            scraped_urls = [url]

    return scraped_urls


def _sync_scrape(
    url,
    output_dir,
    format,
    use_sitemap,
    parallel,
    workers,
    save_chunks,
    chunk_dir,
    progress,
    task,
):
    """Sync scraping implementation."""
    scraper = MarkdownScraper(max_workers=workers)

    if use_sitemap:
        return scraper.scrape_by_sitemap(
            url,
            output_dir=str(output_dir),
            output_format=format,
            parallel=parallel,
            max_workers=workers,
        )
    # Single URL mode - enhanced to support chunks
    html_content = scraper.scrape_website(url)

    # Convert and save
    if scraper.rust_available:
        format_map = {
            "markdown": scraper.OutputFormat.MARKDOWN,
            "json": scraper.OutputFormat.JSON,
            "xml": scraper.OutputFormat.XML,
        }
        output = scraper.convert_html(
            html_content, url, format_map.get(format, scraper.OutputFormat.MARKDOWN)
        )
    else:
        output = scraper._convert_content(html_content, url, format)

    # Save main output
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = "md" if format == "markdown" else format
    output_file = output_dir / f"scraped.{extension}"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)

    # Save chunks if requested
    if save_chunks and format == "markdown":
        chunks = scraper.chunker.create_chunks_from_markdown(output, source_url=url)
        chunks_output_dir = chunk_dir or output_dir
        chunks_output_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks):
            chunk_file = chunks_output_dir / f"scraped_chunk_{i:03d}.md"
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk["content"])

    return [url]


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
    debug: bool = debug_option,
):
    """Convert documents between different formats."""
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
                converted = convert_to_markdown(content)
            elif to_format == "json":
                converted = convert_to_json(content)
            elif to_format == "xml":
                converted = convert_to_xml(content)
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
                    f.write(chunk_data["content"])
            console.print(f"✅ Created {len(chunks)} chunks in {output_dir}")
        else:
            # Save converted content
            if not output_file:
                output_file = input_file.with_suffix(f".{to_format}")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(converted)

            console.print(f"✅ Converted file saved to {output_file}")

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=debug)
        raise typer.Exit(1) from e


@app.command()
def pipeline(
    config_file: Path = typer.Argument(..., help="Pipeline configuration file (YAML)"),
    input_dir: Optional[Path] = typer.Option(
        None, "-i", "--input", help="Input directory (overrides config)"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output directory (overrides config)"
    ),
    debug: bool = debug_option,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be executed without running"
    ),
    continue_on_error: bool = typer.Option(
        False, "--continue", help="Continue pipeline on errors"
    ),
):
    """Execute a RAG pipeline from a configuration file."""
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
                console.print(f"\n  Step {i + 1}: {step.get('name', 'Unnamed step')}")
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
        logger.error(f"Pipeline execution failed: {e}", exc_info=debug)
        raise typer.Exit(1) from e


@app.command()
def chat(
    query: str = typer.Argument(..., help="Query to send to the RAG system"),
    model: str = typer.Option(
        "gpt-4", "-m", "--model", help="Model to use for chat completion"
    ),
    temperature: float = typer.Option(
        0.7, "-t", "--temperature", help="Sampling temperature"
    ),
    debug: bool = debug_option,
):
    """Chat with the RAG system."""
    console.print(Panel.fit("RAGnificent Chat", border_style="blue"))

    try:
        session = RAGChat()
        response = session.chat(query)

        console.print("\n[bold blue]You:[/bold blue]", query)
        console.print("\n[bold green]Assistant:[/bold green]")
        console.print(response)

    except Exception as e:
        logger.error(f"Chat failed: {e}", exc_info=debug)
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
    debug: bool = debug_option,
):
    """Generate embeddings for text content."""
    console.print("🔍 Generating embeddings...")

    try:
        generator = EmbeddingService(model_name=model_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        if input_path.is_file():
            # Process single file
            generator.embed_file(
                input_path, output_dir / f"{input_path.stem}_embeddings.json"
            )
        else:
            # Process directory
            generator.embed_directory(input_path, output_dir, batch_size=batch_size)

        console.print(f"✅ Embeddings saved to {output_dir}")

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=debug)
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
