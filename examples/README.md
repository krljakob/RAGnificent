# RAGnificent Examples

This directory contains usage examples demonstrating various RAGnificent features.

## Quick Start Examples

### 1. Basic Demo (`basic_demo.py`)

Simple demonstration of core scraping and chunking functionality.

```bash
python examples/basic_demo.py
```

### 2. Comprehensive Workflow (`rag_workflow_demo.py`)

Full RAG pipeline with multiple modes and CLI options.

```bash
# Basic mode
python examples/rag_workflow_demo.py --mode basic

# Full workflow
python examples/rag_workflow_demo.py --mode full --urls https://example.com

# Use existing chunks
python examples/rag_workflow_demo.py --mode existing --chunk-files chunks.jsonl
```

## Advanced Examples

### 3. Output Formats (`demo_formats.py`)

Demonstrates all three output formats (Markdown, JSON, XML).

```bash
python examples/demo_formats.py
```

### 4. Custom Configuration (`config_example.py`)

Shows configuration management and customization.

```bash
python examples/config_example.py
```

### 5. Custom Scraper (`custom_scraper_example.py`)

Advanced scraper configuration with sitemap discovery and parallel processing.

```bash
python examples/custom_scraper_example.py
```

### 6. RAG Pipeline (`rag_pipeline_example.py`)

Complete end-to-end RAG implementation with embedding and search.

```bash
python examples/rag_pipeline_example.py
```

## Example Outputs

The `demo_output/` directory contains sample outputs in different formats:

- `output.md` - Markdown format
- `output.json` - JSON format
- `output.xml` - XML format

## Running Examples

All examples can be run from the project root:

```bash
# Make sure dependencies are installed
uv pip install -e .

# Run any example
python examples/<example_name>.py
```
