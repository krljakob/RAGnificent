# RAGnificent Configuration Examples

This directory contains example configuration files for RAGnificent in different formats and for different environments.

## Available Examples

- `default.yaml`: Default configuration settings for all components
- `production.yaml`: Production environment configuration with optimized settings
- `local.json`: Local development configuration in JSON format

## Usage

You can load these configuration files using the enhanced configuration system:

```python
from RAGnificent.core.config import load_config, load_configs_from_directory

# Load a single configuration file
config = load_config("path/to/config/examples/default.yaml")

# Load and merge multiple configuration files from a directory
# Files are loaded in alphabetical order, with later files overriding earlier ones
config = load_configs_from_directory("path/to/config/examples")
```

## Configuration File Formats

RAGnificent supports the following configuration file formats:

- **YAML** (`.yaml`, `.yml`): Human-readable format, good for complex configurations
- **JSON** (`.json`): Machine-readable format, good for programmatic generation
- **Environment Variables** (`.env`): Simple key-value pairs for sensitive information

## Creating Your Own Configuration

You can create your own configuration files based on these examples. The configuration system will validate your settings using Pydantic to ensure they meet the required constraints.

To export your current configuration to a file:

```python
from RAGnificent.core.config import get_config

config = get_config()
config.save_to_file("my_config.yaml")  # or .json
```

## Configuration Hierarchy

When using multiple configuration files, they are merged in the following order:

1. Default settings from code
2. Configuration files (in alphabetical order)
3. Environment variables (highest priority)

This allows for flexible configuration management across different environments.
