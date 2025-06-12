# RAGnificent justfile
# Run commands with 'just <command>'

# Set default environment if not specified
env := "development"

# Default data directory
data_dir := "data"
chunk_dir := data_dir + "/chunks"

# Default output files
raw_documents := data_dir + "/raw_documents.json"
document_chunks := data_dir + "/document_chunks.json"
embedded_chunks := data_dir + "/embedded_chunks.json"

# Default URLs file
links_file := "test_links.txt"

# Default collection name for Qdrant
collection := "ragnificent"

# Show available commands
default:
    @just --list

# Install all Python and Rust dependencies (creates venv if needed)
setup:
    #!/usr/bin/env bash
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment with uv..."
        uv venv
    fi
    export PATH=".venv/bin:$PATH"
    echo "Installing Python dependencies..."
    uv pip install -r requirements.txt
    echo "Building Rust extension with maturin..."
    maturin develop --release

# Build the Rust components with maturin
build:
    maturin build --release
    maturin develop --release

# Build with JavaScript rendering support
build-with-js:
    maturin build --release --features real_rendering
    maturin develop --release --features real_rendering

# Run all Python tests
test-python:
    #!/usr/bin/env bash
    source .venv/bin/activate && pytest

# Run Python binding tests
test-bindings:
    #!/usr/bin/env bash
    source .venv/bin/activate && pytest tests/rust/test_python_bindings.py -v

# Run Rust tests
test-rust:
    cargo test

# Run all tests (Python and Rust)
test: test-rust test-python

# Run all tests with debug logging
test-debug:
    #!/usr/bin/env bash
    RUST_LOG=debug cargo test -- --nocapture
    export PATH=".venv/bin:$PATH"
    pytest -v

# Run benchmarks
bench:
    cargo bench

# Run specific benchmark
bench-html:
    cargo bench html_to_markdown

# Run benchmarks and visualize results
bench-viz:
    #!/usr/bin/env bash
    cargo bench
    export PATH=".venv/bin:$PATH"
    python scripts/visualize_benchmarks.py

# Format Python code
format-python:
    #!/usr/bin/env bash
    export PATH=".venv/bin:$PATH"
    black . --target-version py312
    isort .
    ruff check . --fix

# Format Rust code
format-rust:
    cargo fmt

# Format all code
format: format-rust format-python

# Run linting checks
lint:
    #!/usr/bin/env bash
    cargo clippy
    export PATH=".venv/bin:$PATH"
    ruff check .
    mypy *.py

# Full code quality check with fixes
code-quality:
    #!/usr/bin/env bash
    export PATH=".venv/bin:$PATH"
    black . --target-version py312
    isort .
    ruff check . --fix
    ruff check . --fix --unsafe-fixes
    cargo fmt
    cargo clippy --fix

# Create a clean build for all components
clean-build: clean setup

# Clean build artifacts
clean:
    cargo clean
    rm -rf output/
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    find . -type f -name ".coverage" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name "*.eggs" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +

# One-shot build and test (replaces build_all.sh functionality)
build-all:
    just setup
    just test
    @echo "Build and test complete!"

# Scrape a single URL and convert to markdown
scrape url output="output.md":
    #!/usr/bin/env bash
    source .venv/bin/activate && python -m RAGnificent {{url}} -o {{output}}

# Scrape a single URL and convert to JSON format
scrape-json url output="output.json":
    #!/usr/bin/env bash
    source .venv/bin/activate && python -m RAGnificent {{url}} -o {{output}} -f json

# Scrape a single URL and convert to XML format
scrape-xml url output="output.xml":
    #!/usr/bin/env bash
    source .venv/bin/activate && python -m RAGnificent {{url}} -o {{output}} -f xml

# Scrape a website with sitemap discovery
scrape-sitemap url output_dir="output":
    #!/usr/bin/env bash
    source .venv/bin/activate && python -m RAGnificent {{url}} -o {{output_dir}} --use-sitemap --save-chunks

# Scrape a website with sitemap discovery and parallel processing
scrape-sitemap-parallel url output_dir="output" workers="8":
    #!/usr/bin/env bash
    source .venv/bin/activate && python -m RAGnificent {{url}} -o {{output_dir}} --use-sitemap --save-chunks --parallel --max-workers {{workers}}

# Scrape from a list of URLs in a file
scrape-list output_dir="output" links="test_links.txt":
    python -m RAGnificent -o {{output_dir}} --links-file {{links}}

# Scrape from a list of URLs in parallel
scrape-list-parallel output_dir="output" links="test_links.txt" workers="8":
    python -m RAGnificent -o {{output_dir}} --links-file {{links}} --parallel --max-workers {{workers}}

# Run the complete RAG pipeline with a single URL
rag-pipeline url collection=collection:
    python -c "from RAGnificent.rag.pipeline import Pipeline; pipeline = Pipeline(collection_name='{{collection}}'); pipeline.run_pipeline(url='{{url}}', run_extract=True, run_chunk=True, run_embed=True, run_store=True)"

# Run the complete RAG pipeline with a list of URLs
rag-pipeline-list links=links_file collection=collection:
    python -c "from RAGnificent.rag.pipeline import Pipeline; pipeline = Pipeline(collection_name='{{collection}}'); pipeline.run_pipeline(links_file='{{links}}', run_extract=True, run_chunk=True, run_embed=True, run_store=True)"

# Extract content from URLs
extract url output=raw_documents format="markdown":
    python -c "from RAGnificent.rag.pipeline import Pipeline; Pipeline().extract_content(url='{{url}}', output_file='{{output}}', output_format='{{format}}')"

# Extract content from a list of URLs
extract-list links=links_file output=raw_documents format="markdown":
    python -c "from RAGnificent.rag.pipeline import Pipeline; Pipeline().extract_content(links_file='{{links}}', output_file='{{output}}', output_format='{{format}}')"

# Chunk documents
chunk docs=raw_documents output=document_chunks:
    python -c "from RAGnificent.rag.pipeline import Pipeline; Pipeline().chunk_documents('{{docs}}', '{{output}}')"

# Embed chunks
embed chunks=document_chunks output=embedded_chunks:
    python -c "from RAGnificent.rag.pipeline import Pipeline; Pipeline().embed_chunks('{{chunks}}', '{{output}}')"

# Store chunks in vector database
store chunks=embedded_chunks collection=collection:
    python -c "from RAGnificent.rag.pipeline import Pipeline; Pipeline(collection_name='{{collection}}').store_chunks('{{chunks}}')"

# Search the vector database
search query collection=collection limit="5":
    #!/usr/bin/env bash
    source .venv/bin/activate && python -c "from RAGnificent.rag.pipeline import Pipeline; pipeline = Pipeline(collection_name='{{collection}}'); results = pipeline.search_documents('{{query}}', {{limit}}, as_dict=True); print('\n'.join([f'Score: {r[\"score\"]:.4f}\n{r[\"content\"]}\n' for r in results]))"

# Query with RAG context (requires OpenAI API key)
query query collection=collection limit="3":
    #!/usr/bin/env bash
    source .venv/bin/activate && python -c "from RAGnificent.rag.pipeline import Pipeline; pipeline = Pipeline(collection_name='{{collection}}'); response = pipeline.query_with_context('{{query}}', {{limit}}); print(f'Response: {response[\"response\"]}\n\nSources:\n' + '\n'.join([f'- {r[\"metadata\"][\"document_url\"]}' for r in response['context'] if 'metadata' in r and 'document_url' in r['metadata']]))"

# Run the demo for all output formats
run-demo:
    #!/usr/bin/env bash
    source .venv/bin/activate && python examples/demo_formats.py

# Run a simple example
run-hello:
    #!/usr/bin/env bash
    source .venv/bin/activate && python examples/hello.py

# Run the RAG pipeline example
run-rag-example:
    #!/usr/bin/env bash
    source .venv/bin/activate && python examples/rag_pipeline_example.py

# Visualize Qdrant data
view-qdrant:
    #!/usr/bin/env bash
    source .venv/bin/activate && python view_qdrant_data.py

# Run the web UI
run-webui:
    #!/usr/bin/env bash
    source .venv/bin/activate && python webui.py

# Example end-to-end workflow for a single URL
workflow-single url="https://example.com" collection=collection:
    @echo "Starting end-to-end workflow for {{url}}"
    mkdir -p {{data_dir}} {{chunk_dir}}
    just extract {{url}} {{raw_documents}}
    just chunk {{raw_documents}} {{document_chunks}}
    just embed {{document_chunks}} {{embedded_chunks}}
    just store {{embedded_chunks}} {{collection}}
    @echo "Workflow complete! You can now search the collection:"
    @echo "just search \"your query\" {{collection}}"

# Example end-to-end workflow for a list of URLs
workflow-list links=links_file collection=collection:
    @echo "Starting end-to-end workflow for URLs in {{links}}"
    mkdir -p {{data_dir}} {{chunk_dir}}
    just extract-list {{links}} {{raw_documents}}
    just chunk {{raw_documents}} {{document_chunks}}
    just embed {{document_chunks}} {{embedded_chunks}}
    just store {{embedded_chunks}} {{collection}}
    @echo "Workflow complete! You can now search the collection:"
    @echo "just search \"your query\" {{collection}}"

# Update README.md with latest code/test metrics and badges
update-metrics:
    #!/usr/bin/env bash
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
    python3 scripts/update_readme_metrics.py
