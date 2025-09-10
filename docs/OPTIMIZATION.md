# RAGnificent Performance Optimization

## Performance Architecture

RAGnificent is designed as a hybrid Python/Rust system for optimal performance:

- **Python**: High-level orchestration, API interfaces, and ML operations
- **Rust**: Performance-critical operations like HTML parsing and text processing
- **PyO3 Bindings**: Seamless integration between Python and Rust components

## Optimization Strategies

### 1. Rust Integration

The most performance-critical operations are implemented in Rust:

```rust
// Example: High-performance HTML to Markdown conversion
pub fn convert_html_to_markdown(html: &str, base_url: &str) -> String {
    // Optimized HTML parsing and conversion
    // Up to 10x faster than pure Python implementations
}
```

**Benefits**:

- 5-10x faster HTML parsing and conversion
- Memory-efficient string processing
- Zero-copy operations where possible
- Parallel processing capabilities

### 2. Intelligent Caching

Multi-level caching system reduces redundant operations:

```python
# Two-level cache: Memory + Disk
cache_config = {
    "memory_cache": {
        "max_size": 1000,
        "ttl": 3600
    },
    "disk_cache": {
        "enabled": True,
        "max_size": "1GB",
        "ttl": 86400
    }
}
```

**Cache Layers**:

- **L1 (Memory)**: Fast access to recent requests (LRU eviction)
- **L2 (Disk)**: Persistent cache for larger datasets
- **Embedding Cache**: Avoids recomputing embeddings for identical content

### 3. Parallel Processing

Enable parallel processing for multiple URLs:

```bash
python -m RAGnificent --parallel --max-workers 8 --links-file urls.txt
```

```python
# Programmatic parallel processing
scraper = MarkdownScraper()
scraper.scrape_by_links_file(
    links_file="urls.txt",
    parallel=True,
    max_workers=8
)
```

**Considerations**:

- Optimal worker count: `min(32, (os.cpu_count() or 1) + 4)`
- Respect rate limits: Total rate = rate_limit / num_workers
- Memory usage scales with worker count

### 4. Memory Management

Efficient memory usage through resource management:

```python
# Resource manager configuration
resource_config = {
    "max_memory": "2GB",
    "connection_pool_size": 100,
    "cleanup_interval": 300,
    "gc_threshold": 0.8
}
```

**Memory Optimizations**:

- **Connection Pooling**: Reuse HTTP connections
- **Streaming Processing**: Process large documents in chunks
- **Garbage Collection**: Proactive cleanup of unused objects
- **Memory Monitoring**: Track and limit memory usage

### 5. Rate Limiting and Throttling

Respectful and efficient web scraping:

```python
# Adaptive rate limiting
throttler = RequestThrottler(
    requests_per_second=1.0,
    adaptive=True,
    backoff_strategy="exponential"
)
```

**Features**:

- **Adaptive Rate Limiting**: Adjusts based on server responses
- **Exponential Backoff**: Handles rate limit errors gracefully
- **Domain-Specific Limits**: Different limits for different domains
- **Circuit Breaker**: Temporarily stops requests to failing servers

## Benchmarking Results

### HTML to Markdown Conversion

| Implementation | Speed | Memory |
|---------------|-------|---------|
| Pure Python | 1.0x | 1.0x |
| Rust (single-threaded) | 5.2x | 0.6x |
| Rust (multi-threaded) | 8.7x | 0.7x |

### Chunking Performance

| Document Size | Time (Python) | Time (Rust) | Speedup |
|--------------|---------------|-------------|---------|
| 10KB | 12ms | 3ms | 4x |
| 100KB | 89ms | 15ms | 6x |
| 1MB | 1.2s | 180ms | 6.7x |

### Embedding Generation

| Batch Size | Time per Item | Throughput |
|-----------|---------------|------------|
| 1 | 45ms | 22 items/sec |
| 8 | 12ms | 83 items/sec |
| 32 | 8ms | 125 items/sec |
| 128 | 7ms | 143 items/sec |

## Performance Tuning

### 1. Chunk Size Optimization

Balance between context and performance:

```python
# For speed-optimized chunking
config = {
    "chunk_size": 500,      # Smaller chunks = faster processing
    "chunk_overlap": 50,    # Less overlap = less redundancy
    "strategy": "recursive" # Fastest chunking strategy
}

# For quality-optimized chunking
config = {
    "chunk_size": 1000,     # Larger chunks = better context
    "chunk_overlap": 200,   # More overlap = better continuity
    "strategy": "semantic"  # Best quality chunking
}
```

### 2. Embedding Optimization

Optimize embedding generation:

```python
# Fast embedding configuration
embedding_config = {
    "model_name": "all-MiniLM-L6-v2",  # Smaller, faster model
    "batch_size": 64,                   # Larger batches
    "max_length": 256,                  # Shorter sequences
    "normalize": False                  # Skip normalization
}

# Quality embedding configuration
embedding_config = {
    "model_name": "BAAI/bge-large-en-v1.5",  # Larger, better model
    "batch_size": 16,                          # Smaller batches for stability
    "max_length": 512,                         # Full context
    "normalize": True                          # Normalized embeddings
}
```

### 3. Vector Store Optimization

Optimize vector database operations:

```python
# Production vector store setup
vector_config = {
    "host": "dedicated-qdrant-server",
    "collection_config": {
        "vectors": {
            "size": 384,
            "distance": "Cosine"
        },
        "optimizers_config": {
            "default_segment_number": 2,
            "max_segment_size": 200000
        },
        "hnsw_config": {
            "m": 16,
            "ef_construct": 100,
            "full_scan_threshold": 10000
        }
    }
}
```

### 4. JavaScript Rendering Optimization

When using JavaScript rendering feature:

```python
# Optimized JS rendering
js_config = {
    "headless": True,
    "disable_images": True,        # Skip image loading
    "disable_css": True,           # Skip CSS loading
    "timeout": 10000,              # 10 second timeout
    "wait_for": "networkidle",     # Wait strategy
    "viewport": {
        "width": 1024,
        "height": 768
    }
}
```

## Monitoring and Profiling

### 1. Built-in Statistics

RAGnificent includes performance monitoring:

```python
from RAGnificent.core.stats import StatsMixin

class MyProcessor(StatsMixin):
    def process(self):
        with self.timer("processing"):
            # Your processing code
            pass
        
        # View statistics
        stats = self.get_stats()
        print(f"Processing time: {stats['processing']['mean']:.2f}s")
```

### 2. Memory Profiling

Monitor memory usage:

```python
from RAGnificent.core.resource_manager import ResourceManager

# Enable memory monitoring
resource_manager = ResourceManager(
    enable_monitoring=True,
    memory_limit="2GB"
)

# Check memory usage
memory_info = resource_manager.get_memory_info()
print(f"Memory usage: {memory_info['used_mb']:.1f}MB")
```

### 3. Performance Benchmarks

Run comprehensive benchmarks:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmarks
cargo bench html_to_markdown
cargo bench chunk_markdown
cargo bench parallel_processing

# Visualize results
python scripts/visualize_benchmarks.py
```

## Production Optimization Checklist

### Infrastructure

- [ ] Use dedicated Qdrant server (not in-memory)
- [ ] Enable connection pooling
- [ ] Configure appropriate rate limits
- [ ] Set up monitoring and alerting
- [ ] Use SSD storage for caching

### Configuration

- [ ] Optimize chunk sizes for your use case
- [ ] Choose appropriate embedding model
- [ ] Configure cache sizes based on available memory
- [ ] Enable parallel processing with appropriate worker count
- [ ] Set conservative rate limits for production

### Monitoring

- [ ] Track scraping latency and success rates
- [ ] Monitor embedding generation performance
- [ ] Watch memory usage and garbage collection
- [ ] Monitor vector database performance
- [ ] Set up alerts for performance degradation

### Scaling

- [ ] Implement horizontal scaling for high-volume workloads
- [ ] Use separate embedding service for GPU acceleration
- [ ] Implement caching layer (Redis) for frequent queries
- [ ] Consider content deduplication for efficiency
- [ ] Plan for vector database scaling

## Common Performance Issues

### Issue: Slow HTML Processing

**Solutions**:

- Ensure Rust components are compiled with `--release`
- Use parallel processing for multiple URLs
- Increase chunk sizes to reduce overhead

### Issue: High Memory Usage

**Solutions**:

- Reduce cache sizes
- Use streaming processing for large documents
- Enable garbage collection optimizations
- Limit parallel worker count

### Issue: Slow Embedding Generation

**Solutions**:

- Use smaller, faster models for development
- Increase batch sizes (if memory allows)
- Implement embedding caching
- Consider GPU acceleration for production

### Issue: Vector Search Latency

**Solutions**:

- Optimize Qdrant configuration (HNSW parameters)
- Use appropriate vector dimensions
- Implement search result caching
- Consider approximate search for large datasets

## Advanced Optimizations

### 1. Custom Rust Extensions

For specialized use cases, implement custom Rust functions:

```rust
#[pyfunction]
fn custom_html_processor(html: &str, options: PyDict) -> PyResult<String> {
    // Custom high-performance processing
    Ok(processed_content)
}
```

### 2. Streaming Processing

For very large documents:

```python
def stream_process_large_document(url: str):
    """Process large documents in streaming fashion"""
    for chunk in scraper.stream_content(url, chunk_size=10000):
        processed_chunk = process_chunk(chunk)
        yield processed_chunk
```

### 3. Content Deduplication

Avoid processing duplicate content:

```python
from RAGnificent.core.cache import ContentHashCache

hash_cache = ContentHashCache()
content_hash = hash_cache.get_content_hash(content)
if not hash_cache.is_processed(content_hash):
    # Process new content
    result = process_content(content)
    hash_cache.mark_processed(content_hash, result)
```
