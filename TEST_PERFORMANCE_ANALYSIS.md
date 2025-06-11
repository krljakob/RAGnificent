# Test Performance Analysis for RAGnificent

## Summary of Performance Issues

After analyzing the test suite, I've identified several potential causes of slowness:

### 1. **Performance Benchmarks Running by Default**

- **Location**: `/tests/benchmarks/test_performance.py`
- **Issue**: The `pyproject.toml` has benchmark settings enabled by default (`benchmark_enable = true`)
- **Impact**: Performance tests with intentional delays are running during regular test execution
- **Examples**:
  - `time.sleep(0.01 * len(chunks))` in mock services
  - `time.sleep(0.05)` simulating network delays
  - `time.sleep(0.1)` in parallel scraping tests
  - `time.sleep(2)` in worker timeout tests

### 2. **Heavy Model Loading During Tests**

- **Location**: Tests importing `EmbeddingService`
- **Issue**: SentenceTransformer models are loaded in `__init__` method
- **Impact**: Loading transformer models (even small ones like `all-MiniLM-L6-v2`) takes significant time
- **Affected tests**: Any test that instantiates `EmbeddingService` or `Pipeline`

### 3. **Integration Tests Without Proper Mocking**

- **Location**: `/tests/integration/test_rag_integration.py`
- **Issue**: Integration tests may be loading real models and performing actual embeddings
- **Impact**: Each test run loads ML models and performs computations

### 4. **I/O Operations in Tests**

- **Location**: Multiple test files
- **Issue**: Tests creating temporary directories, files, and performing disk I/O
- **Impact**: Disk operations add overhead, especially on slower storage

## Recommendations

### 1. **Disable Benchmarks by Default**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
benchmark_disable = true  # Change from false to true
# Or add markers to skip benchmarks by default
markers = [
    "benchmark: marks tests as benchmarks (deselect with '-m \"not benchmark\"')",
]
```

### 2. **Create pytest.ini for Test Configuration**

Create a `pytest.ini` file:

```ini
[pytest]
# Skip benchmark tests by default
addopts = -m "not benchmark"
# Add markers
markers =
    benchmark: marks tests as benchmarks
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

### 3. **Mark Slow Tests Appropriately**

Add markers to slow tests:

```python
@pytest.mark.benchmark
def test_cache_performance():
    ...

@pytest.mark.slow
@pytest.mark.integration
def test_end_to_end_pipeline():
    ...
```

### 4. **Mock Heavy Resources**

Create fixtures that mock heavy resources:

```python
@pytest.fixture
def mock_embedding_service(monkeypatch):
    """Mock embedding service to avoid loading models."""
    mock_service = MagicMock()
    mock_service.embed_chunks.return_value = [
        {"content": chunk["content"], "embedding": [0.1] * 384} 
        for chunk in chunks
    ]
    monkeypatch.setattr("RAGnificent.rag.embedding.EmbeddingService", mock_service)
    return mock_service
```

### 5. **Separate Test Commands**

Add to documentation:

```bash
# Run only fast unit tests
pytest -m "not slow and not benchmark and not integration"

# Run all tests including slow ones
pytest

# Run only benchmarks
pytest -m benchmark

# Run with specific timeout
pytest --timeout=10
```

### 6. **Add Test Profiling**

To identify slow tests:

```bash
# Show slowest 10 tests
pytest --durations=10

# Profile test execution
pytest --profile
```

## Immediate Actions

1. **Quick Fix** - Run tests without benchmarks:

   ```bash
   pytest -m "not benchmark" tests/
   ```

2. **Add pytest configuration** to skip slow tests by default

3. **Mock ML model loading** in unit tests

4. **Separate integration tests** that require real resources

5. **Add timeout** to prevent hanging tests:

   ```bash
   pytest --timeout=30
   ```

## Specific Slow Test Locations

1. **Performance Benchmarks** (`/tests/benchmarks/test_performance.py`):
   - `test_cache_performance`: 100+ cache operations
   - `test_throttler_performance`: Multiple sleep calls
   - `test_chunker_performance`: Processing large content
   - `test_pipeline_performance`: Full pipeline with delays
   - `test_parallel_scraping_performance`: Network simulation delays

2. **Error Handling Tests** (`/tests/unit/test_scraper_error_handling.py`):
   - `test_worker_timeout_handling`: Contains `time.sleep(2)`
   - Multiple retry attempts with delays

3. **Integration Tests** (`/tests/integration/test_rag_integration.py`):
   - Loading actual ML models
   - Performing real embeddings
   - Vector store operations
