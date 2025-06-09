# RAGnificent Configuration

## Configuration System

RAGnificent uses a hierarchical configuration system with Pydantic settings that supports multiple configuration sources and environment-specific overrides.

## Configuration Sources

Configuration is loaded in the following order (later sources override earlier ones):

1. **Default values** (hardcoded in the application)
2. **Configuration files** (YAML or JSON)
3. **Environment variables**
4. **Command-line arguments**

## Configuration Files

### Directory Structure

```
config/
├── environments/
│   ├── development.yaml
│   ├── production.yaml
│   └── testing.yaml
└── examples/
    ├── default.yaml
    ├── local.json
    └── production.yaml
```

### Environment-Specific Configs

Configuration files can be organized by environment:

- `config/environments/development.yaml` - Development settings
- `config/environments/production.yaml` - Production settings  
- `config/environments/testing.yaml` - Test environment settings

### Example Configuration Files

#### Default Configuration (YAML)

```yaml
# config/examples/default.yaml
scraper:
  requests_per_second: 1.0
  chunk_size: 1000
  chunk_overlap: 200
  max_retries: 3
  timeout: 30
  
chunking:
  strategy: "semantic"  # Options: recursive, semantic, sliding_window
  max_chunk_size: 1000
  overlap_size: 200
  preserve_headers: true
  
embedding:
  model_type: "sentence_transformer"
  model_name: "BAAI/bge-small-en-v1.5"
  cache_enabled: true
  batch_size: 32
  
vector_store:
  host: ":memory:"
  port: 6333
  collection_name: "ragnificent"
  https: false
  
cache:
  enabled: true
  max_size: 1000
  ttl: 3600
  disk_cache_enabled: true
  
feature_flags:
  advanced_chunking: false
  parallel_processing: true
  memory_optimization: true
  security_features: true
```

#### Production Configuration (YAML)

```yaml
# config/environments/production.yaml
scraper:
  requests_per_second: 0.5  # More conservative in production
  max_retries: 5
  timeout: 60
  
vector_store:
  host: "your-qdrant-server.com"
  port: 6333
  https: true
  api_key: "${QDRANT_API_KEY}"  # Environment variable
  
cache:
  max_size: 10000
  ttl: 86400  # 24 hours
  
logging:
  level: "INFO"
  format: "json"
  file: "/var/log/ragnificent/app.log"
  
feature_flags:
  advanced_chunking: true
  memory_optimization: true
```

#### Local Development (JSON)

```json
{
  "scraper": {
    "requests_per_second": 2.0,
    "chunk_size": 500
  },
  "embedding": {
    "model_name": "all-MiniLM-L6-v2"
  },
  "vector_store": {
    "host": "localhost",
    "port": 6333
  },
  "feature_flags": {
    "advanced_chunking": true,
    "parallel_processing": false
  }
}
```

## Environment Variables

All configuration values can be overridden using environment variables with the prefix `RAGNIFICENT_`:

```bash
# Scraper settings
export RAGNIFICENT_SCRAPER_REQUESTS_PER_SECOND=0.5
export RAGNIFICENT_SCRAPER_CHUNK_SIZE=800

# Embedding settings
export RAGNIFICENT_EMBEDDING_MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
export RAGNIFICENT_EMBEDDING_CACHE_ENABLED=true

# Vector store settings
export RAGNIFICENT_VECTOR_STORE_HOST="production-qdrant.example.com"
export RAGNIFICENT_VECTOR_STORE_API_KEY="your-secret-key"

# Feature flags
export RAGNIFICENT_FEATURE_FLAGS_ADVANCED_CHUNKING=true
export RAGNIFICENT_FEATURE_FLAGS_PARALLEL_PROCESSING=false
```

## Configuration Loading

### Programmatic Usage

```python
from RAGnificent.core.config import load_config, AppConfig

# Load default configuration
config = load_config()

# Load specific environment configuration
config = load_config(environment="production")

# Load from specific file
config = load_config(config_file="config/my_config.yaml")

# Access configuration values
print(f"Requests per second: {config.scraper.requests_per_second}")
print(f"Chunk size: {config.scraper.chunk_size}")
```

### Command Line Usage

```bash
# Use default configuration
python -m RAGnificent https://example.com

# Specify configuration file
python -m RAGnificent https://example.com --config config/production.yaml

# Override specific values
python -m RAGnificent https://example.com --requests-per-second 0.5 --chunk-size 800
```

## Configuration Sections

### Scraper Configuration

```yaml
scraper:
  requests_per_second: 1.0      # Rate limiting
  chunk_size: 1000              # Default chunk size
  chunk_overlap: 200            # Chunk overlap
  max_retries: 3                # Retry attempts
  timeout: 30                   # Request timeout
  user_agent: "RAGnificent/1.0" # Custom user agent
  follow_redirects: true        # Follow HTTP redirects
  verify_ssl: true              # SSL certificate verification
```

### Chunking Configuration

```yaml
chunking:
  strategy: "semantic"          # Chunking strategy
  max_chunk_size: 1000         # Maximum chunk size
  overlap_size: 200            # Overlap between chunks
  preserve_headers: true       # Keep header structure
  min_chunk_size: 100          # Minimum chunk size
  header_separators:           # Custom header separators
    - "# "
    - "## "
    - "### "
```

### Embedding Configuration

```yaml
embedding:
  model_type: "sentence_transformer"  # Model type
  model_name: "BAAI/bge-small-en-v1.5" # Model name
  cache_enabled: true                 # Enable embedding cache
  batch_size: 32                      # Batch size for processing
  max_length: 512                     # Maximum sequence length
  normalize_embeddings: true          # Normalize embedding vectors
```

### Vector Store Configuration

```yaml
vector_store:
  host: ":memory:"              # Qdrant host (:memory: for in-memory)
  port: 6333                    # Qdrant port
  collection_name: "ragnificent" # Collection name
  https: false                  # Use HTTPS
  api_key: null                 # API key for authentication
  timeout: 60                   # Connection timeout
  prefer_grpc: false            # Use gRPC instead of HTTP
```

### Cache Configuration

```yaml
cache:
  enabled: true                 # Enable caching
  max_size: 1000               # Maximum number of cached items
  ttl: 3600                    # Time to live (seconds)
  disk_cache_enabled: true     # Enable disk cache
  disk_cache_dir: ".cache"     # Disk cache directory
```

### Feature Flags

```yaml
feature_flags:
  advanced_chunking: false     # Enable experimental chunking
  parallel_processing: true    # Enable parallel URL processing
  memory_optimization: true    # Enable memory optimizations
  security_features: true      # Enable security features
  js_rendering: false          # Enable JavaScript rendering
  rollout_percentage: 100      # Percentage rollout for gradual deployment
```

### Logging Configuration

```yaml
logging:
  level: "INFO"                # Log level (DEBUG, INFO, WARNING, ERROR)
  format: "text"               # Log format (text, json)
  file: null                   # Log file path (null for stdout)
  max_file_size: "10MB"        # Maximum log file size
  backup_count: 5              # Number of backup files
```

## Validation

The configuration system includes comprehensive validation:

- **Type checking**: Ensures values are of the correct type
- **Range validation**: Validates numeric ranges (e.g., requests_per_second > 0)
- **Enum validation**: Validates enumerated values (e.g., chunking strategies)
- **URL validation**: Validates URL formats and accessibility
- **File path validation**: Ensures file paths are valid and accessible

## Configuration Best Practices

### Development
- Use higher rate limits for faster development
- Enable debug logging
- Use smaller chunk sizes for faster testing
- Keep feature flags enabled for testing new features

### Production
- Use conservative rate limits to be respectful to target sites
- Configure external vector database instead of in-memory
- Enable comprehensive logging with log rotation
- Use environment variables for sensitive configuration (API keys)
- Implement monitoring for configuration changes

### Security
- Never commit API keys or sensitive data to configuration files
- Use environment variables or secure vaults for secrets
- Validate all configuration inputs
- Implement proper access controls for configuration files