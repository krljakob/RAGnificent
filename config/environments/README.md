# RAGnificent Environment Configurations

This directory contains environment-specific configuration files for RAGnificent. These configurations allow you to run the application with different settings optimized for various environments.

## Available Environments

- **Development (`development.yaml`)**: Optimized for local development with debugging enabled
- **Testing (`testing.yaml`)**: Configured for automated testing and CI/CD pipelines
- **Production (`production.yaml`)**: Optimized for production deployment with performance and security in mind

## Usage

To use a specific environment configuration, you can load it using the `load_config` function:

```python
from RAGnificent.core.config import load_config

# Load development configuration
config = load_config("config/environments/development.yaml")

# Use the configuration
pipeline = Pipeline(
    collection_name=config.qdrant.collection,
    embedding_model_type=config.embedding.model_type,
    embedding_model_name=config.embedding.model_name,
)
```

You can also load and merge multiple configuration files from a directory using the `load_configs_from_directory` function:

```python
from RAGnificent.core.config import load_configs_from_directory

# Load and merge all configurations in the directory
# Files are loaded in alphabetical order, with later files overriding earlier ones
config = load_configs_from_directory("config/environments")
```

## Environment Variables

All configuration settings can also be overridden using environment variables. For example:

```bash
# Set the logging level to DEBUG
export LOGGING_LEVEL=DEBUG

# Enable request caching
export SCRAPER_CACHE_ENABLED=true

# Set the embedding model
export EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
```

## Feature Flags

Feature flags allow you to enable or disable specific features at runtime. You can configure them in the environment configuration files under the `features` section:

```yaml
# Feature flags
features:
  enable_advanced_chunking: true
  enable_parallel_processing: true
  enable_memory_optimization: true
  enable_caching: true
  enable_benchmarking: false
  enable_security_features: true
```

You can also set feature flags using environment variables:

```bash
# Enable experimental embeddings
export RAGNIFICENT_FEATURE_EXPERIMENTAL_EMBEDDINGS=true
```

## Resource Management

Resource management settings control how the application manages system resources like memory and connections:

```yaml
# Resource management
resources:
  max_memory_percent: 75.0
  max_connections: 200
  max_thread_workers: 50
  cleanup_interval: 30
  enable_monitoring: true
```

## Customizing Configurations

You can create your own environment configurations by copying one of the existing files and modifying it to suit your needs. For example:

```bash
cp config/environments/development.yaml config/environments/staging.yaml
```

Then edit `staging.yaml` to customize the settings for your staging environment.

## Configuration Hierarchy

When multiple configuration sources are available, they are applied in the following order (later sources override earlier ones):

1. Default values defined in the code
2. Configuration files
3. Environment variables

This allows you to have a base configuration and override specific settings as needed.
