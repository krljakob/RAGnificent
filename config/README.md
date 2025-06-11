# RAGnificent Configuration

This directory contains configuration files for different environments and usage scenarios.

## Directory Structure

- `environments/` - Environment-specific configurations (development, testing, production)
- `examples/` - Example configuration files in different formats

## Quick Usage

```python
from RAGnificent.core.config import load_config

# Load environment config
config = load_config("config/environments/production.yaml")

# Load example config
config = load_config("config/examples/default.yaml")
```

## Configuration Formats

Supported formats:

- **YAML** (`.yaml`, `.yml`) - Human-readable, recommended
- **JSON** (`.json`) - Machine-readable, programmatic use

## Environment Variables

All settings can be overridden with environment variables using the `RAGNIFICENT_` prefix:

```bash
export RAGNIFICENT_SCRAPER_REQUESTS_PER_SECOND=0.5
export RAGNIFICENT_EMBEDDING_MODEL_NAME="BAAI/bge-large-en-v1.5"
```

For detailed configuration options, see [Configuration Documentation](../docs/CONFIGURATION.md).
