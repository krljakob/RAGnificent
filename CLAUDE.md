# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Commands & Guidelines for RAGnificent

## Build & Test Commands

### Environment Setup
- `uv venv` - Create virtual environment
- `uv pip install -r requirements.txt` - Install Python dependencies
- `uv pip install -e .` - Install package in editable mode
- `export PATH=".venv/bin:$PATH"` - Activate virtual environment (Unix/macOS)

### Building & Testing
- `cargo build` - Build Rust components
- `cargo build --release --features real_rendering` - Build with JS rendering support
- `pytest` - Run all Python tests (requires proper environment setup)
- `pytest tests/rust/test_python_bindings.py -v` - Run Python binding tests
- `pytest tests/unit/test_chunk_utils.py -v` - Run chunk utilities tests
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

## Justfile Commands (Alternative Interface)

This project includes a comprehensive `justfile` for common development tasks:

- `just setup` - Install Python and Rust dependencies
- `just build` - Build Rust components
- `just build-with-js` - Build with JavaScript rendering support
- `just test` - Run all tests (Python and Rust)
- `just test-debug` - Run tests with debug logging
- `just bench` - Run all benchmarks
- `just bench-viz` - Run benchmarks and visualize results
- `just format` - Format all code (Python and Rust)
- `just lint` - Run linting checks
- `just code-quality` - Full code quality check with fixes
- `just clean` - Clean build artifacts
- `just scrape <url>` - Quick single URL scraping
- `just scrape-sitemap <url>` - Scrape with sitemap discovery
- `just workflow-single <url>` - Complete end-to-end RAG workflow

## Development Dependencies

- **uv**: Package installer and dependency manager (preferred over pip)
- **maturin**: Build backend for Python/Rust hybrid projects
- **pytest**: Testing framework with benchmark support
- **ruff**: Fast Python linter and formatter
- **black**: Python code formatter
- **mypy**: Static type checker for Python
- **cargo**: Rust package manager and build tool

## Configuration System

The project uses a hierarchical configuration system with Pydantic settings:
- `config/environments/` - Environment-specific configs (development, production, testing)
- `config/examples/` - Example configuration files
- Configuration can be loaded from YAML or JSON files
- Environment variables override file-based settings

## Module Import Strategy & Test Environment

### Package Installation (Recommended)
The preferred approach is to install the package in editable mode:
```bash
uv pip install -e .
export PATH=".venv/bin:$PATH"
pytest
```

This allows imports like:
```python
from RAGnificent.core.cache import RequestCache
from RAGnificent.utils.chunk_utils import ContentChunker
```

### Current Test Status
- **90+ tests** currently working and collected (significantly improved from previous 64)
- **Major test fixes completed**: Fixed 22 failing tests including performance benchmarks, config tests, embedding edge cases, and nested header chunking
- **Working test categories**: chunk utils, main functionality, Rust bindings, benchmarks, sitemap utils, embedding service, scraper error handling, pipeline tests, search tests
- **Recent improvements**: Fixed method signatures, parameter passing, mock configurations, and test expectations to match actual implementation

### Fallback Import Strategy
Some modules use fallback imports for compatibility:
```python
try:
    from .stats import StatsMixin
except ImportError:
    from stats import StatsMixin
```

### For New Tests
Follow the package-based import pattern and ensure the virtual environment is properly activated with the package installed in editable mode.

## Test Fixes Applied

### Performance Test Fixes
- **Throttler tests**: Fixed URL parameter passing in execute methods and mock functions
- **Chunker tests**: Added missing `source_url` parameter to `create_chunks_from_markdown` calls
- **Pipeline tests**: Updated `MockVectorStore.store_documents` to accept required keyword arguments
- **Parallel scraping tests**: Fixed lambda function signatures and sitemap URL object creation

### Configuration Test Fixes
- **ChunkingStrategy enum**: Changed invalid `SENTENCE` value to correct `SEMANTIC` value
- **Test expectations**: Updated assertions to match available enum values

### Embedding Test Fixes
- **Mock patching**: Updated mock decorator paths from module-level to actual import paths
- **OpenAI mocks**: Changed from `RAGnificent.rag.embedding.openai` to `openai.embeddings.create`
- **SentenceTransformer mocks**: Changed from module-level to `sentence_transformers.SentenceTransformer`

### Nested Header Chunking Fixes
- **Test expectations**: Updated tests to check for content patterns rather than non-existent header hierarchy
- **Section parsing**: Adjusted expected section counts and header levels to match actual test data
- **Content matching**: Changed from heading_path checks to content-based chunk identification

### Best Practices for Test Development
- Use package-based imports (`from RAGnificent.module import Class`)
- Match mock paths to actual import locations
- Verify test data contains expected content before writing assertions
- Use correct enum values and method signatures from the actual codebase
- Test with realistic mock objects that accept expected parameters
