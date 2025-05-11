"""
Performance benchmarking tests for RAGnificent.

These tests measure the performance of key operations in the RAG pipeline
to identify bottlenecks and validate optimizations.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from RAGnificent.core.cache import RequestCache
from RAGnificent.core.config import get_config
from RAGnificent.core.scraper import MarkdownScraper
from RAGnificent.core.throttle import RequestThrottler
from RAGnificent.rag.embedding import get_embedding_service
from RAGnificent.rag.pipeline import Pipeline
from RAGnificent.rag.vector_store import get_vector_store
from RAGnificent.utils.chunk_utils import ContentChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmarks")


class PerformanceTimer:
    """Utility class for timing operations."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.duration = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"BENCHMARK - {self.name}: {self.duration:.4f} seconds")


class MemoryMonitor:
    """Utility class for monitoring memory usage."""

    def __init__(self, name: str):
        self.name = name
        self.start_memory = 0
        self.end_memory = 0
        self.memory_delta = 0

    def __enter__(self):
        import psutil
        self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import psutil
        self.end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        self.memory_delta = self.end_memory - self.start_memory
        logger.info(f"MEMORY - {self.name}: {self.memory_delta:.2f} MB delta, {self.end_memory:.2f} MB total")


@pytest.fixture
def test_urls():
    """Sample URLs for benchmarking."""
    return [
        "https://www.python.org/",
        "https://docs.python.org/3/tutorial/",
        "https://docs.python.org/3/library/",
        "https://docs.python.org/3/reference/",
        "https://docs.python.org/3/howto/",
    ]


@pytest.fixture
def test_documents():
    """Sample documents for benchmarking."""
    return [
        {
            "url": "https://www.python.org/",
            "title": "Python Programming Language",
            "content": "Python is a programming language that lets you work quickly and integrate systems more effectively.",
            "timestamp": "2023-01-01T00:00:00Z",
        },
        {
            "url": "https://docs.python.org/3/tutorial/",
            "title": "Python Tutorial",
            "content": "This tutorial introduces the reader to the basic concepts and features of the Python language and system.",
            "timestamp": "2023-01-01T00:00:00Z",
        },
        {
            "url": "https://docs.python.org/3/library/",
            "title": "Python Standard Library",
            "content": "The Python Standard Library contains built-in modules that provide access to system functionality.",
            "timestamp": "2023-01-01T00:00:00Z",
        },
        {
            "url": "https://docs.python.org/3/reference/",
            "title": "Python Language Reference",
            "content": "This reference manual describes the syntax and core semantics of the language.",
            "timestamp": "2023-01-01T00:00:00Z",
        },
        {
            "url": "https://docs.python.org/3/howto/",
            "title": "Python HOWTOs",
            "content": "Python HOWTOs are documents that cover a single specific topic.",
            "timestamp": "2023-01-01T00:00:00Z",
        },
    ]


@pytest.fixture
def test_chunks():
    """Sample chunks for benchmarking."""
    return [
        {
            "url": "https://www.python.org/",
            "title": "Python Programming Language",
            "content": "Python is a programming language that lets you work quickly.",
            "chunk_id": "1",
            "heading_path": ["Python Programming Language"],
        },
        {
            "url": "https://www.python.org/",
            "title": "Python Programming Language",
            "content": "Python helps integrate systems more effectively.",
            "chunk_id": "2",
            "heading_path": ["Python Programming Language"],
        },
        {
            "url": "https://docs.python.org/3/tutorial/",
            "title": "Python Tutorial",
            "content": "This tutorial introduces the reader to the basic concepts of Python.",
            "chunk_id": "3",
            "heading_path": ["Python Tutorial"],
        },
        {
            "url": "https://docs.python.org/3/tutorial/",
            "title": "Python Tutorial",
            "content": "The tutorial covers features of the Python language and system.",
            "chunk_id": "4",
            "heading_path": ["Python Tutorial"],
        },
        {
            "url": "https://docs.python.org/3/library/",
            "title": "Python Standard Library",
            "content": "The Python Standard Library contains built-in modules.",
            "chunk_id": "5",
            "heading_path": ["Python Standard Library"],
        },
        {
            "url": "https://docs.python.org/3/library/",
            "title": "Python Standard Library",
            "content": "These modules provide access to system functionality.",
            "chunk_id": "6",
            "heading_path": ["Python Standard Library"],
        },
        {
            "url": "https://docs.python.org/3/reference/",
            "title": "Python Language Reference",
            "content": "This reference manual describes the syntax of the language.",
            "chunk_id": "7",
            "heading_path": ["Python Language Reference"],
        },
        {
            "url": "https://docs.python.org/3/reference/",
            "title": "Python Language Reference",
            "content": "The manual also covers core semantics of the language.",
            "chunk_id": "8",
            "heading_path": ["Python Language Reference"],
        },
        {
            "url": "https://docs.python.org/3/howto/",
            "title": "Python HOWTOs",
            "content": "Python HOWTOs are documents that cover a single specific topic.",
            "chunk_id": "9",
            "heading_path": ["Python HOWTOs"],
        },
        {
            "url": "https://docs.python.org/3/howto/",
            "title": "Python HOWTOs",
            "content": "These documents provide detailed information on specific Python features.",
            "chunk_id": "10",
            "heading_path": ["Python HOWTOs"],
        },
    ]


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for benchmarking."""
    class MockEmbeddingService:
        def embed_chunks(self, chunks):
            time.sleep(0.01 * len(chunks))
            for chunk in chunks:
                chunk["embedding"] = [0.1] * 384  # Mock embedding vector
            return chunks

    return MockEmbeddingService()


@pytest.fixture
def mock_vector_store():
    """Mock vector store for benchmarking."""
    class MockVectorStore:
        def __init__(self):
            self.documents = []

        def store_documents(self, documents):
            time.sleep(0.01 * len(documents))
            self.documents.extend(documents)
            return True

        def search(self, query_vector, limit=5, threshold=0.7):
            time.sleep(0.02)
            return self.documents[:min(limit, len(self.documents))]

        def count_documents(self):
            return len(self.documents)

    return MockVectorStore()


def test_cache_performance():
    """Benchmark the performance of the RequestCache."""
    cache_dir = Path("./test_cache")
    cache_dir.mkdir(exist_ok=True)

    with PerformanceTimer("Cache initialization (default)"):
        cache = RequestCache(cache_dir=str(cache_dir))

    with PerformanceTimer("Cache set (100 items)"):
        for i in range(100):
            url = f"https://example.com/page{i}"
            content = f"Content for page {i}" * 100  # Make content reasonably sized
            cache.set(url, content)

    with PerformanceTimer("Cache get (100 hits)"):
        for i in range(100):
            url = f"https://example.com/page{i}"
            content = cache.get(url)
            assert content is not None

    with PerformanceTimer("Cache get (100 misses)"):
        for i in range(100, 200):
            url = f"https://example.com/page{i}"
            content = cache.get(url)
            assert content is None

    with PerformanceTimer("Cache with compression"):
        compressed_cache = RequestCache(
            cache_dir=str(cache_dir) + "_compressed",
            compression_threshold=100,  # Set low to force compression
        )
        for i in range(50):
            url = f"https://example.com/compressed{i}"
            content = f"Compressed content for page {i}" * 200  # Large content
            compressed_cache.set(url, content)

    with PerformanceTimer("Cache clear"):
        count = cache.clear()
        logger.info(f"Cleared {count} cache entries")

    with PerformanceTimer("Cache pattern invalidation"):
        for i in range(50):
            url = f"https://example.com/pattern{i}"
            content = f"Pattern content for page {i}"
            cache.set(url, content)
        
        count = cache.invalidate(pattern="pattern[0-9]+")
        logger.info(f"Invalidated {count} cache entries with pattern")

    with PerformanceTimer("Cache statistics"):
        stats = cache.get_stats()
        logger.info(f"Cache stats: {stats}")

    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)
    shutil.rmtree(cache_dir.with_name(cache_dir.name + "_compressed"), ignore_errors=True)


def test_throttler_performance():
    """Benchmark the performance of the RequestThrottler."""
    with PerformanceTimer("Throttler initialization (default)"):
        throttler = RequestThrottler()

    with PerformanceTimer("Throttler with domain limits"):
        domain_limits = {
            "example.com": 2.0,
            "test.com": 0.5,
            "*.python.org": 5.0,
        }
        throttler = RequestThrottler(
            requests_per_second=1.0,
            domain_specific_limits=domain_limits,
            max_workers=20,
            adaptive_throttling=True,
        )

    with PerformanceTimer("Throttle (10 requests, same domain)"):
        for i in range(10):
            throttler.throttle("https://example.com/page")

    with PerformanceTimer("Throttle (10 requests, different domains)"):
        domains = ["example.com", "test.com", "python.org", "docs.python.org"]
        for i in range(10):
            domain = domains[i % len(domains)]
            throttler.throttle(f"https://{domain}/page")

    def mock_request(url):
        time.sleep(0.05)  # Simulate network delay
        return type("Response", (), {"status_code": 200})

    with PerformanceTimer("Execute (5 requests)"):
        for i in range(5):
            result = throttler.execute(mock_request, f"https://example.com/page{i}")

    with PerformanceTimer("Execute parallel (20 requests)"):
        urls = [f"https://example.com/page{i}" for i in range(20)]
        results = throttler.execute_parallel(mock_request, urls)
        assert len(results) == 20

    with PerformanceTimer("Throttler statistics"):
        stats = throttler.get_stats()
        logger.info(f"Throttler stats: {stats}")


def test_chunker_performance():
    """Benchmark the performance of the ContentChunker."""
    small_content = "# Heading 1\nThis is a paragraph.\n## Heading 2\nAnother paragraph."
    medium_content = "\n".join([f"# Heading {i}\nParagraph {i}.\n## Subheading {i}.1\nMore text.\n### Subheading {i}.1.1\nEven more text." for i in range(1, 11)])
    large_content = "\n".join([f"# Heading {i}\nParagraph {i}.\n## Subheading {i}.1\nMore text.\n### Subheading {i}.1.1\nEven more text." for i in range(1, 101)])

    for chunk_size, chunk_overlap in [(200, 50), (500, 100), (1000, 200)]:
        with PerformanceTimer(f"Chunker initialization (size={chunk_size}, overlap={chunk_overlap})"):
            chunker = ContentChunker(chunk_size, chunk_overlap)

        with PerformanceTimer(f"Chunking small content (size={chunk_size})"):
            small_chunks = chunker.create_chunks_from_markdown(small_content)
            logger.info(f"Created {len(small_chunks)} chunks from small content")

        with PerformanceTimer(f"Chunking medium content (size={chunk_size})"):
            medium_chunks = chunker.create_chunks_from_markdown(medium_content)
            logger.info(f"Created {len(medium_chunks)} chunks from medium content")

        with PerformanceTimer(f"Chunking large content (size={chunk_size})"):
            large_chunks = chunker.create_chunks_from_markdown(large_content)
            logger.info(f"Created {len(large_chunks)} chunks from large content")

    from RAGnificent.core.config import ChunkingStrategy
    
    for strategy in [ChunkingStrategy.SEMANTIC, ChunkingStrategy.SLIDING_WINDOW, ChunkingStrategy.RECURSIVE]:
        with PerformanceTimer(f"Chunking with {strategy.name} strategy"):
            chunker = ContentChunker(500, 100)
            if strategy == ChunkingStrategy.SEMANTIC:
                chunks = chunker.create_chunks_from_markdown(medium_content)
            elif strategy == ChunkingStrategy.SLIDING_WINDOW:
                chunks = chunker.create_sliding_window_chunks(medium_content)
            elif strategy == ChunkingStrategy.RECURSIVE:
                chunks = chunker.create_recursive_chunks(medium_content)
            
            logger.info(f"Created {len(chunks)} chunks with {strategy.name} strategy")


def test_pipeline_performance(test_documents, test_chunks, mock_embedding_service, mock_vector_store):
    """Benchmark the performance of the RAG Pipeline."""
    with PerformanceTimer("Pipeline initialization"):
        pipeline = Pipeline(
            collection_name="benchmark_collection",
            embedding_model_type="sentence_transformer",
            embedding_model_name="all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=100,
            requests_per_second=5.0,
            cache_enabled=True,
        )
        pipeline.embedding_service = mock_embedding_service
        pipeline.vector_store = mock_vector_store

    with PerformanceTimer("Pipeline chunking"):
        with MemoryMonitor("Pipeline chunking"):
            chunks = pipeline.chunk_documents(test_documents)
            logger.info(f"Created {len(chunks)} chunks from {len(test_documents)} documents")

    with PerformanceTimer("Pipeline embedding"):
        with MemoryMonitor("Pipeline embedding"):
            embedded_chunks = pipeline.embed_chunks(test_chunks)
            logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

    with PerformanceTimer("Pipeline storage"):
        with MemoryMonitor("Pipeline storage"):
            success = pipeline.store_chunks(embedded_chunks)
            assert success
            logger.info(f"Stored {len(embedded_chunks)} chunks in vector store")

    with PerformanceTimer("Pipeline search"):
        with MemoryMonitor("Pipeline search"):
            results = pipeline.search_documents("Python programming", limit=5)
            logger.info(f"Found {len(results)} results for search query")

    with PerformanceTimer("Complete pipeline with backpressure"):
        with MemoryMonitor("Complete pipeline"):
            result = pipeline.run_pipeline(
                urls=[doc["url"] for doc in test_documents],
                output_format="markdown",
                batch_size=2,  # Small batch size to test backpressure
                enable_backpressure=True,
                enable_benchmarking=True,
            )
            logger.info(f"Pipeline result: {result}")
            if "benchmarks" in result:
                for step, metrics in result["benchmarks"].items():
                    logger.info(f"Pipeline benchmark - {step}: {metrics}")


def test_parallel_scraping_performance(test_urls):
    """Benchmark the performance of parallel scraping."""
    with PerformanceTimer("Scraper initialization (default)"):
        scraper = MarkdownScraper()

    with PerformanceTimer("Scraper with enhanced parallel processing"):
        enhanced_scraper = MarkdownScraper(
            requests_per_second=5.0,
            domain_specific_limits={"python.org": 10.0},
            max_workers=10,
            adaptive_throttling=True,
        )

    def mock_scrape_website(url):
        time.sleep(0.1)  # Simulate network delay
        return f"<html><body><h1>Title for {url}</h1><p>Content for {url}</p></body></html>"

    enhanced_scraper.scrape_website = mock_scrape_website

    with PerformanceTimer("Sequential scraping (5 URLs)"):
        results = []
        for url in test_urls:
            html = enhanced_scraper.scrape_website(url)
            results.append(html)
        logger.info(f"Scraped {len(results)} URLs sequentially")

    with PerformanceTimer("Parallel scraping with sitemap (5 URLs)"):
        original_discover = enhanced_scraper._discover_urls_from_sitemap
        enhanced_scraper._discover_urls_from_sitemap = lambda url: test_urls

        original_process = enhanced_scraper._process_single_url
        enhanced_scraper._process_single_url = lambda url, *args, **kwargs: {"url": url, "content": f"Content for {url}"}

        results = enhanced_scraper.scrape_by_sitemap("https://example.com/sitemap.xml", limit=5)
        logger.info(f"Scraped {len(results)} URLs in parallel from sitemap")

        enhanced_scraper._discover_urls_from_sitemap = original_discover
        enhanced_scraper._process_single_url = original_process

    with PerformanceTimer("Parallel scraping with links file (5 URLs)"):
        links_file = Path("./test_links.txt")
        with open(links_file, "w") as f:
            for url in test_urls:
                f.write(f"{url}\n")

        enhanced_scraper._process_single_url = lambda url, *args, **kwargs: {"url": url, "content": f"Content for {url}"}

        results = enhanced_scraper.scrape_by_links_file(str(links_file), limit=5)
        logger.info(f"Scraped {len(results)} URLs in parallel from links file")

        links_file.unlink()


if __name__ == "__main__":
    test_cache_performance()
    test_throttler_performance()
    test_chunker_performance()
    test_pipeline_performance(test_documents, test_chunks, mock_embedding_service, mock_vector_store)
    test_parallel_scraping_performance(test_urls)
