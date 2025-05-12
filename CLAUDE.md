# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Commands & Guidelines for RAGnificent

## Build & Test Commands

- `cargo build` - Build Rust components
- `cargo build --release --features real_rendering` - Build with JS rendering support
- `uv pip install -r requirements.txt` - Install Python dependencies
- `pytest` - Run all Python tests
- `pytest tests/test_python_bindings.py -v` - Run Python binding tests
- `pytest test_main.py::test_convert_to_markdown -v` - Run specific Python test
- `pytest test_main.py::test_format_conversion -v` - Test JSON and XML output formats
- `cargo test` - Run Rust tests
- `RUST_LOG=debug cargo test -- --nocapture` - Run Rust tests with logging
- `cargo bench` - Run all benchmarks
- `cargo bench html_to_markdown` - Run specific benchmark
- `python demo_formats.py` - Demonstrate all output formats (markdown, JSON, XML)
- `./build_all.sh` - One-shot build & test (Unix/macOS)
- `.\build_all.ps1` - One-shot build & test (Windows PowerShell)
- `cargo bench && python scripts/visualize_benchmarks.py` - Run benchmarks and visualize results

## Code Quality Commands

- `ruff check . --fix` - Run linter and auto-fix issues
- `ruff check . --fix --unsafe-fixes` - Run linter with more aggressive fixes
- `black .` - Format Python code
- `isort .` - Sort imports
- `sourcery review . --fix` - Analyze and improve code quality
- `mypy *.py` - Type checking

## Project Architecture

RAGnificent is a hybrid Python/Rust project for web scraping and content processing, specializing in converting HTML to various formats (Markdown, JSON, XML) with support for semantic chunking for RAG (Retrieval-Augmented Generation) applications.

### Key Components

1. **Core Python Components**:
   - `core/scraper.py`: Main scraper implementation with request handling
   - `core/cache.py`: Two-level caching system (memory/disk) for HTTP requests
   - `core/throttle.py`: Rate limiting for respectful web scraping
   - `core/config.py`: Configuration management

2. **Rust Components**:
   - `src/lib.rs`: Main entry point and PyO3 bindings
   - `src/markdown_converter.rs`: HTML to Markdown/JSON/XML conversion
   - `src/chunker.rs`: Semantic chunking implementation
   - `src/js_renderer.rs`: JavaScript page rendering (optional feature)

3. **Utility Modules**:
   - `utils/chunk_utils.py`: Semantic chunking utilities for RAG
   - `utils/sitemap_utils.py`: Sitemap parsing for systematic scraping

4. **RAG Components**:
   - `rag/embedding.py`: Text embedding functionality
   - `rag/vector_store.py`: Vector storage for embeddings
   - `rag/search.py`: Vector search implementation
   - `rag/pipeline.py`: End-to-end RAG pipeline

### Data Flow

1. Web content is scraped via `MarkdownScraper` in `core/scraper.py`
2. HTML is processed and converted to the desired format (Markdown/JSON/XML)
3. Content is optionally chunked for RAG applications
4. Chunks can be embedded and stored in a vector database
5. Content can be searched using semantic similarity

## Code Style Guidelines

- **Python**: Python 3.12+ with type annotations
- **Imports**: Group imports (stdlib, third-party, local)
- **Formatting**: Follow PEP 8 guidelines
- **Error handling**: Use exception handling with specific exceptions
- **Naming**: snake_case for Python, snake_case for Rust
- **Testing**: Use pytest fixtures and mocks
- **Documentation**: Docstrings for public functions and classes
- **Rust**: Follow Rust 2024 edition idioms and use thiserror for errors
- **Type annotations**: Required for all new code

## Output Format Features

- **Markdown**: Human-readable plain text format (default)
- **JSON**: Structured data format for programmatic usage
  - Document structure with title, headers, paragraphs, links, images, etc.
  - Serialized with proper indentation for readability
- **XML**: Markup format for document interchange
  - Document structure with proper XML tags and hierarchy
  - Includes XML declaration and proper escaping
- Supported HTML elements:
  - Headers (h1-h6)
  - Paragraphs
  - Links (with resolved relative URLs)
  - Images (with resolved relative URLs)
  - Ordered and unordered lists
  - Blockquotes
  - Code blocks
- Use `-f/--format` CLI argument to specify output format

## Command Line Arguments

- `-f, --format` : Output format (markdown, json, xml)
- `--save-chunks` : Save content chunks for RAG
- `--chunk-dir <dir>` : Directory to save chunks
- `--parallel` : Enable parallel URL processing
- `--max-workers <n>` : Number of parallel workers
- `--use-sitemap` : Use sitemap.xml to discover URLs
- `--min-priority` : Minimum priority for sitemap URLs
- `--include` : Regex patterns for URLs to include
- `--exclude` : Regex patterns for URLs to exclude
- `--chunk-size` : Maximum chunk size (chars)
- `--chunk-overlap` : Overlap between chunks (chars)
