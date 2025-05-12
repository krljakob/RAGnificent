# RAGnificent Makefile
# Run commands with 'make <command>'

# Set default environment
ENV ?= development

# Default directories
DATA_DIR ?= data
CHUNK_DIR ?= $(DATA_DIR)/chunks

# Default output files
RAW_DOCUMENTS ?= $(DATA_DIR)/raw_documents.json
DOCUMENT_CHUNKS ?= $(DATA_DIR)/document_chunks.json
EMBEDDED_CHUNKS ?= $(DATA_DIR)/embedded_chunks.json

# Default URLs file
LINKS_FILE ?= links.txt

# Default collection name for Qdrant
COLLECTION ?= ragnificent

# Default output format
FORMAT ?= markdown

# Default number of workers for parallel processing
WORKERS ?= 8

# Default search limit
LIMIT ?= 5

# Install all Python and Rust dependencies
.PHONY: setup
setup:
	uv pip install -r requirements.txt
	cargo build --release

# Build the Rust components
.PHONY: build
build:
	cargo build --release

# Build with JavaScript rendering support
.PHONY: build-with-js
build-with-js:
	cargo build --release --features real_rendering

# Run all Python tests
.PHONY: test-python
test-python:
	pytest

# Run Python binding tests
.PHONY: test-bindings
test-bindings:
	pytest tests/rust/test_python_bindings.py -v

# Run Rust tests
.PHONY: test-rust
test-rust:
	cargo test

# Run all tests (Python and Rust)
.PHONY: test
test: test-rust test-python

# Run all tests with debug logging
.PHONY: test-debug
test-debug:
	RUST_LOG=debug cargo test -- --nocapture
	pytest -v

# Run benchmarks
.PHONY: bench
bench:
	cargo bench

# Run specific benchmark
.PHONY: bench-html
bench-html:
	cargo bench html_to_markdown

# Run benchmarks and visualize results
.PHONY: bench-viz
bench-viz:
	cargo bench
	python scripts/visualize_benchmarks.py

# Format Python code
.PHONY: format-python
format-python:
	black .
	isort .
	ruff check . --fix

# Format Rust code
.PHONY: format-rust
format-rust:
	cargo fmt

# Format all code
.PHONY: format
format: format-rust format-python

# Run linting checks
.PHONY: lint
lint:
	cargo clippy
	ruff check .
	mypy *.py

# Full code quality check with fixes
.PHONY: code-quality
code-quality:
	black .
	isort .
	ruff check . --fix
	sourcery review . --fix
	cargo fmt
	cargo clippy --fix

# Create a clean build for all components
.PHONY: clean-build
clean-build: clean build

# Clean build artifacts
.PHONY: clean
clean:
	cargo clean
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.eggs" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# One-shot build and test for Unix-like systems
.PHONY: build-all-unix
build-all-unix:
	./build_all.sh

# One-shot build and test for Windows
.PHONY: build-all-windows
build-all-windows:
	powershell -File .\build_all.ps1

# Make data directories
.PHONY: mkdirs
mkdirs:
	mkdir -p $(DATA_DIR) $(CHUNK_DIR)

# Scrape a single URL and convert to specified format
.PHONY: scrape
scrape:
ifeq ($(URL),)
	@echo "Error: URL is required. Use 'make scrape URL=https://example.com'"
	@exit 1
endif
	python -m RAGnificent $(URL) -o $(OUTPUT) -f $(FORMAT)

# Scrape a website with sitemap discovery
.PHONY: scrape-sitemap
scrape-sitemap:
ifeq ($(URL),)
	@echo "Error: URL is required. Use 'make scrape-sitemap URL=https://example.com'"
	@exit 1
endif
	python -m RAGnificent $(URL) -o $(OUTPUT_DIR) --use-sitemap --save-chunks

# Scrape a website with sitemap discovery and parallel processing
.PHONY: scrape-sitemap-parallel
scrape-sitemap-parallel:
ifeq ($(URL),)
	@echo "Error: URL is required. Use 'make scrape-sitemap-parallel URL=https://example.com'"
	@exit 1
endif
	python -m RAGnificent $(URL) -o $(OUTPUT_DIR) --use-sitemap --save-chunks --parallel --max-workers $(WORKERS)

# Scrape from a list of URLs in a file
.PHONY: scrape-list
scrape-list:
	python -m RAGnificent -o $(OUTPUT_DIR) --links-file $(LINKS_FILE)

# Scrape from a list of URLs in parallel
.PHONY: scrape-list-parallel
scrape-list-parallel:
	python -m RAGnificent -o $(OUTPUT_DIR) --links-file $(LINKS_FILE) --parallel --max-workers $(WORKERS)

# Run the complete RAG uv pipeline with a single URL
.PHONY: rag-uv pipeline
rag-uv pipeline:
ifeq ($(URL),)
	@echo "Error: URL is required. Use 'make rag-uv pipeline URL=https://example.com'"
	@exit 1
endif
	python -c "from RAGnificent.rag.uv pipeline import uv pipeline; uv pipeline = uv pipeline(collection_name='$(COLLECTION)'); uv pipeline.run_uv pipeline(url='$(URL)', run_extract=True, run_chunk=True, run_embed=True, run_store=True)"

# Run the complete RAG uv pipeline with a list of URLs
.PHONY: rag-uv pipeline-list
rag-uv pipeline-list:
	python -c "from RAGnificent.rag.uv pipeline import uv pipeline; uv pipeline = uv pipeline(collection_name='$(COLLECTION)'); uv pipeline.run_uv pipeline(links_file='$(LINKS_FILE)', run_extract=True, run_chunk=True, run_embed=True, run_store=True)"

# Extract content from a URL
.PHONY: extract
extract:
ifeq ($(URL),)
	@echo "Error: URL is required. Use 'make extract URL=https://example.com'"
	@exit 1
endif
	python -c "from RAGnificent.rag.uv pipeline import uv pipeline; uv pipeline().extract_content(url='$(URL)', output_file='$(RAW_DOCUMENTS)', output_format='$(FORMAT)')"

# Extract content from a list of URLs
.PHONY: extract-list
extract-list:
	python -c "from RAGnificent.rag.uv pipeline import uv pipeline; uv pipeline().extract_content(links_file='$(LINKS_FILE)', output_file='$(RAW_DOCUMENTS)', output_format='$(FORMAT)')"

# Chunk documents
.PHONY: chunk
chunk:
	python -c "from RAGnificent.rag.uv pipeline import uv pipeline; uv pipeline().chunk_documents('$(RAW_DOCUMENTS)', '$(DOCUMENT_CHUNKS)')"

# Embed chunks
.PHONY: embed
embed:
	python -c "from RAGnificent.rag.uv pipeline import uv pipeline; uv pipeline().embed_chunks('$(DOCUMENT_CHUNKS)', '$(EMBEDDED_CHUNKS)')"

# Store chunks in vector database
.PHONY: store
store:
	python -c "from RAGnificent.rag.uv pipeline import uv pipeline; uv pipeline(collection_name='$(COLLECTION)').store_chunks('$(EMBEDDED_CHUNKS)')"

# Search the vector database
.PHONY: search
search:
ifeq ($(QUERY),)
	@echo "Error: QUERY is required. Use 'make search QUERY=\"your query\"'"
	@exit 1
endif
	python -c "from RAGnificent.rag.search import search; results = search('$(QUERY)', $(LIMIT), collection_name='$(COLLECTION)'); print('\n'.join([f'Score: {r[\"score\"]:.4f}\n{r[\"content\"]}\n' for r in results]))"

# Query with RAG context (requires OpenAI API key)
.PHONY: query
query:
ifeq ($(QUERY),)
	@echo "Error: QUERY is required. Use 'make query QUERY=\"your query\"'"
	@exit 1
endif
	python -c "from RAGnificent.rag.uv pipeline import uv pipeline; response = uv pipeline(collection_name='$(COLLECTION)').query_with_context('$(QUERY)', $(LIMIT)); print(f'Response: {response[\"response\"]}\n\nSources:\n' + '\n'.join([f'- {r[\"source_url\"]}' for r in response['context']]))"

# Run the demo for all output formats
.PHONY: run-demo
run-demo:
	python examples/demo_formats.py

# Run a simple example
.PHONY: run-hello
run-hello:
	python examples/hello.py

# Run the RAG uv pipeline example
.PHONY: run-rag-example
run-rag-example:
	python examples/rag_uv pipeline_example.py

# Visualize Qdrant data
.PHONY: view-qdrant
view-qdrant:
	python view_qdrant_data.py

# Run the web UI
.PHONY: run-webui
run-webui:
	python webui.py

# Example end-to-end workflow for a single URL
.PHONY: workflow-single
workflow-single:
ifeq ($(URL),)
	@echo "Error: URL is required. Use 'make workflow-single URL=https://example.com'"
	@exit 1
endif
	@echo "Starting end-to-end workflow for $(URL)"
	@make mkdirs
	@make extract URL=$(URL)
	@make chunk
	@make embed
	@make store
	@echo "Workflow complete! You can now search the collection:"
	@echo "make search QUERY=\"your query\" COLLECTION=$(COLLECTION)"

# Example end-to-end workflow for a list of URLs
.PHONY: workflow-list
workflow-list:
	@echo "Starting end-to-end workflow for URLs in $(LINKS_FILE)"
	@make mkdirs
	@make extract-list
	@make chunk
	@make embed
	@make store
	@echo "Workflow complete! You can now search the collection:"
	@echo "make search QUERY=\"your query\" COLLECTION=$(COLLECTION)"

# Help command
.PHONY: help
help:
	@echo "RAGnificent Makefile Help"
	@echo ""
	@echo "Basic Commands:"
	@echo "  make setup                    - Install all dependencies"
	@echo "  make build                    - Build Rust components"
	@echo "  make test                     - Run all tests"
	@echo "  make format                   - Format all code"
	@echo "  make lint                     - Run linting checks"
	@echo "  make clean                    - Clean build artifacts"
	@echo ""
	@echo "Scraping Commands:"
	@echo "  make scrape URL=<url>         - Scrape a single URL"
	@echo "  make scrape-sitemap URL=<url> - Scrape a website with sitemap"
	@echo "  make scrape-list              - Scrape from links.txt"
	@echo ""
	@echo "RAG uv pipeline Commands:"
	@echo "  make workflow-single URL=<url> - Run complete workflow for single URL"
	@echo "  make workflow-list             - Run complete workflow for URLs in links.txt"
	@echo "  make extract URL=<url>         - Extract content from URL"
	@echo "  make chunk                     - Chunk documents"
	@echo "  make embed                     - Embed chunks"
	@echo "  make store                     - Store in vector database"
	@echo "  make search QUERY=\"query\"      - Search vector database"
	@echo ""
	@echo "Examples:"
	@echo "  make run-demo                 - Run demo of output formats"
	@echo "  make run-hello                - Run simple example"
	@echo "  make run-rag-example          - Run RAG uv pipeline example"
	@echo ""
	@echo "For more information, see README.md"
