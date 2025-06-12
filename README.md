![RAGnificent](docs/assets/github-banner.svg)

# RAGnificent 🔄📝

[![Lines of Code](https://img.shields.io/badge/lines%20of%20code-9,779-blue?logo=github)](https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)
[![Test Coverage](https://img.shields.io/badge/coverage-unknown-brightgreen?logo=pytest)](https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)
[![Test Count](https://img.shields.io/badge/tests-unknown-yellow?logo=pytest)](https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)(https://github.com/krljakob/RAGnificent)

RAGnificent combines Python and Rust components to scrape websites and convert HTML content to markdown, JSON, or XML formats. It supports sitemap parsing, semantic chunking for RAG (Retrieval-Augmented Generation), and includes performance optimizations through Rust integration.

Key features include HTML-to-markdown/JSON/XML conversion with support for various elements, intelligent content chunking that preserves document structure, and systematic content discovery through sitemap parsing. The hybrid architecture uses Python for high-level operations and Rust for performance-critical tasks.

> **Note:** Codebase metrics (lines of code, test count, coverage) are updated automatically via CI or scripts. See the [Automation section](#automation--metrics) below for details.

Check out the [deepwiki](https://deepwiki.com/krljakob/RAGnificent/) for a granular breakdown of the repository contents, purpose and structure.

## Documentation

- 📖 **[Features](docs/FEATURES.md)** - Comprehensive feature overview and capabilities
- ⚙️ **[Configuration](docs/CONFIGURATION.md)** - Configuration management and environment setup
- 🚀 **[Optimization](docs/OPTIMIZATION.md)** - Performance tuning and optimization guide

## Installation

```bash
git clone https://github.com/krljakob/RAGnificent.git
cd RAGnificent

# Quick setup
./build_all.sh  # Unix/macOS
# or: .\build_all.ps1  # Windows

# Manual setup
uv venv && export PATH=".venv/bin:$PATH"
uv pip install -r requirements.txt && uv pip install -e .
pytest
```

## Quick Start

```bash
# Basic conversion
python -m RAGnificent https://example.com -o output.md

# With RAG chunking
python -m RAGnificent https://example.com --save-chunks --chunk-dir chunks

# Multiple formats and parallel processing
python -m RAGnificent --links-file urls.txt --parallel -f json
```

```python
# Python API
from RAGnificent.core.scraper import MarkdownScraper

scraper = MarkdownScraper()
html = scraper.scrape_website("https://example.com")
markdown = scraper.convert_to_markdown(html, "https://example.com")
chunks = scraper.create_chunks(markdown, "https://example.com")
```

## Testing

```bash
# Run all tests (including benchmarks)
pytest

# Fast development testing (recommended)
./run_tests.sh fast  # ~15 seconds
# or
pytest -m "not benchmark and not slow"

# Run specific test categories
./run_tests.sh unit         # Unit tests only
./run_tests.sh integration  # Integration tests
./run_tests.sh benchmark    # Performance benchmarks
./run_tests.sh profile      # Show slowest tests

# Run specific test files
pytest tests/unit/test_chunk_utils.py -v
pytest tests/rust/test_python_bindings.py -v
```

**Test Performance**: Tests are organized by speed - fast unit tests run in ~15 seconds, while full suite including benchmarks takes ~22 seconds. Benchmarks are skipped by default for rapid development cycles.

## Automation & Metrics

- **Lines of Code**: Calculated automatically using [`cloc`](https://github.com/AlDanial/cloc) or [`tokei`](https://github.com/XAMPPRocky/tokei). See the badge above for the latest count.
- **Test Coverage & Count**: Updated via [`pytest-cov`](https://pytest-cov.readthedocs.io/) and CI. See the badges above for live stats.
- **How to Update**: Run `just update-metrics` or use the provided CI workflow. See `scripts/update_readme_metrics.py` for details.
- **Performance Best Practices**: See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) for up-to-date performance strategies and benchmarking results.

## Development

### Code Organization

- `RAGnificent/`: Main Python package
  - `core/`: Core functionality (scraper, cache, config, etc.)
  - `rag/`: RAG-specific components (embedding, vector store, search)
  - `utils/`: Utility modules (chunking, sitemap parsing)
- `src/`: Rust source code for performance-critical operations
- `tests/`: Comprehensive test suite
- `examples/`: Demo scripts and usage examples
- `docs/`: Detailed documentation

### Running Benchmarks

```bash
cargo bench
python scripts/visualize_benchmarks.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE file](LICENSE) for details.

## Author

🐍🦀 krljakob
