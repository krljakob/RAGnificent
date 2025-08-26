"""
End-to-end integration tests for the complete RAG workflow.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.integration
class TestE2ERAGWorkflow:
    """Test complete RAG workflow from scraping to search."""

    @pytest.fixture
    def mock_html_content(self):
        """Sample HTML content for testing."""
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Introduction to RAG</h1>
            <p>Retrieval-Augmented Generation (RAG) is a powerful technique that combines 
            the strengths of retrieval systems with generative models.</p>
            
            <h2>Key Components</h2>
            <p>RAG systems consist of three main components:</p>
            <ul>
                <li>Document chunking and preprocessing</li>
                <li>Embedding generation and vector storage</li>
                <li>Semantic search and retrieval</li>
            </ul>
            
            <h2>Implementation</h2>
            <p>Modern RAG implementations use transformer-based models for embedding generation
            and vector databases for efficient similarity search.</p>
        </body>
        </html>
        """

    @pytest.fixture
    def mock_scraper(self, mock_html_content):
        """Mock scraper that returns test content."""
        with patch("RAGnificent.core.scraper.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = mock_html_content
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_get.return_value = mock_response
            
            from RAGnificent.core.scraper import MarkdownScraper
            yield MarkdownScraper(
                requests_per_second=10,
                cache_enabled=False
            )

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service that returns consistent embeddings."""
        with patch("RAGnificent.rag.embedding.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            
            # Create deterministic embeddings based on text hash
            def encode_side_effect(texts, *args, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                embeddings = []
                for text in texts:
                    # Generate deterministic embedding based on text length
                    np.random.seed(len(text))
                    embedding = np.random.rand(384)
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]
            
            mock_model.encode.side_effect = encode_side_effect
            mock_st.return_value = mock_model
            
            from RAGnificent.rag.embedding import get_embedding_service
            yield get_embedding_service("sentence-transformers")

    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        from RAGnificent.rag.vector_store import VectorStore
        
        class MockVectorStore(VectorStore):
            def __init__(self, *args, **kwargs):
                self.documents = []
                self.embeddings = []
                
            def store_documents(self, documents, embeddings, **kwargs):
                self.documents.extend(documents)
                self.embeddings.extend(embeddings)
                return len(documents)
                
            def search(self, query_embedding, top_k=5, **kwargs):
                if not self.embeddings:
                    return []
                
                # Simple cosine similarity search
                similarities = []
                for i, emb in enumerate(self.embeddings):
                    similarity = np.dot(query_embedding, emb)
                    similarities.append((similarity, i))
                
                similarities.sort(reverse=True)
                results = []
                for score, idx in similarities[:top_k]:
                    results.append({
                        "content": self.documents[idx],
                        "score": score,
                        "metadata": {"index": idx}
                    })
                return results
        
        with patch("RAGnificent.rag.vector_store.get_vector_store") as mock_get:
            mock_get.return_value = MockVectorStore()
            yield mock_get.return_value

    def test_complete_workflow(self, mock_scraper, mock_embedding_service, mock_vector_store):
        """Test the complete RAG workflow from scraping to search."""
        # Step 1: Scrape content
        url = "https://example.com/test"
        markdown_content = mock_scraper.scrape_website(url)
        
        assert markdown_content is not None
        assert "Introduction to RAG" in markdown_content
        assert "Retrieval-Augmented Generation" in markdown_content
        
        # Step 2: Chunk content
        from RAGnificent.utils.chunk_utils import ContentChunker
        
        chunker = ContentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.create_chunks_from_markdown(
            markdown_content,
            source_url=url
        )
        
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        
        # Step 3: Generate embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = mock_embedding_service.embed(texts)
        
        assert len(embeddings) == len(chunks)
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        
        # Step 4: Store in vector database
        stored_count = mock_vector_store.store_documents(texts, embeddings)
        assert stored_count == len(chunks)
        
        # Step 5: Search for content
        query = "What are the main components of RAG?"
        query_embedding = mock_embedding_service.embed(query)
        
        search_results = mock_vector_store.search(query_embedding, top_k=3)
        
        assert len(search_results) <= 3
        assert all("content" in result for result in search_results)
        assert all("score" in result for result in search_results)
        
        # Verify search relevance
        top_result = search_results[0] if search_results else None
        assert top_result is not None
        assert "components" in top_result["content"].lower() or \
               "RAG" in top_result["content"]

    def test_pipeline_class_integration(self, mock_scraper, mock_embedding_service, mock_vector_store):
        """Test the Pipeline class for end-to-end workflow."""
        with patch("RAGnificent.rag.pipeline.MarkdownScraper") as mock_scraper_class:
            mock_scraper_class.return_value = mock_scraper
            
            with patch("RAGnificent.rag.pipeline.get_embedding_service") as mock_emb_service:
                mock_emb_service.return_value = mock_embedding_service
                
                with patch("RAGnificent.rag.pipeline.get_vector_store") as mock_vs:
                    mock_vs.return_value = mock_vector_store
                    
                    from RAGnificent.rag.pipeline import Pipeline
                    
                    pipeline = Pipeline(
                        collection_name="test_collection",
                        chunk_size=200,
                        chunk_overlap=50,
                        cache_enabled=False
                    )
                    
                    # Process URL
                    urls = ["https://example.com/test"]
                    result = pipeline.process_urls(urls)
                    
                    assert result["urls_processed"] == 1
                    assert result["chunks_created"] > 0
                    assert result["embeddings_generated"] > 0
                    
                    # Search
                    search_results = pipeline.search("RAG components", top_k=3)
                    assert len(search_results) > 0

    def test_multiple_format_outputs(self, mock_scraper):
        """Test scraping with different output formats."""
        url = "https://example.com/test"
        
        # Test markdown output
        markdown = mock_scraper.scrape_website(url, output_format="markdown")
        assert "# Introduction to RAG" in markdown
        
        # Test JSON output
        json_output = mock_scraper.scrape_website(url, output_format="json")
        parsed = json.loads(json_output)
        assert "title" in parsed
        assert "headers" in parsed
        assert len(parsed["headers"]) > 0
        
        # Test XML output
        xml_output = mock_scraper.scrape_website(url, output_format="xml")
        assert "<?xml version" in xml_output
        assert "<document>" in xml_output
        assert "<title>" in xml_output

    def test_error_handling_in_workflow(self, mock_vector_store):
        """Test error handling throughout the RAG workflow."""
        from RAGnificent.rag.pipeline import Pipeline
        
        with patch("RAGnificent.rag.pipeline.MarkdownScraper") as mock_scraper_class:
            # Simulate scraping failure
            mock_scraper_class.return_value.scrape_website.side_effect = Exception("Network error")
            
            pipeline = Pipeline(
                collection_name="test_collection",
                continue_on_error=True
            )
            
            # Should handle error gracefully
            result = pipeline.process_urls(["https://example.com/fail"])
            assert result["urls_processed"] == 0
            assert result["errors"] == 1

    def test_batch_processing(self, mock_scraper, mock_embedding_service, mock_vector_store):
        """Test batch processing of multiple URLs."""
        from RAGnificent.rag.pipeline import Pipeline
        
        with patch("RAGnificent.rag.pipeline.MarkdownScraper") as mock_scraper_class:
            mock_scraper_class.return_value = mock_scraper
            
            with patch("RAGnificent.rag.pipeline.get_embedding_service") as mock_emb_service:
                mock_emb_service.return_value = mock_embedding_service
                
                with patch("RAGnificent.rag.pipeline.get_vector_store") as mock_vs:
                    mock_vs.return_value = mock_vector_store
                    
                    pipeline = Pipeline(
                        collection_name="test_batch",
                        chunk_size=200,
                        chunk_overlap=50,
                        cache_enabled=False
                    )
                    
                    # Process multiple URLs
                    urls = [
                        "https://example.com/doc1",
                        "https://example.com/doc2",
                        "https://example.com/doc3"
                    ]
                    
                    result = pipeline.process_urls(urls, parallel=True, max_workers=3)
                    
                    assert result["urls_processed"] == 3
                    assert result["chunks_created"] > 0
                    assert result["embeddings_generated"] > 0

    def test_chunking_strategies(self, mock_html_content):
        """Test different chunking strategies."""
        from RAGnificent.utils.chunk_utils import ContentChunker
        
        # Convert HTML to markdown first
        from markdownify import markdownify
        markdown = markdownify(mock_html_content)
        
        # Test recursive chunking
        chunker = ContentChunker(chunk_size=100, chunk_overlap=20)
        recursive_chunks = chunker.create_chunks_from_markdown(
            markdown,
            source_url="https://example.com",
            chunking_strategy="recursive"
        )
        
        assert len(recursive_chunks) > 0
        assert all(len(chunk["content"]) <= 120 for chunk in recursive_chunks)  # Allow for overlap
        
        # Test semantic chunking
        semantic_chunks = chunker.create_chunks_from_markdown(
            markdown,
            source_url="https://example.com",
            chunking_strategy="semantic"
        )
        
        assert len(semantic_chunks) > 0
        # Semantic chunks should preserve logical sections
        
        # Test sliding window
        sliding_chunks = chunker.create_chunks_from_markdown(
            markdown,
            source_url="https://example.com",
            chunking_strategy="sliding_window"
        )
        
        assert len(sliding_chunks) > 0

    @pytest.mark.skipif(not Path(".venv").exists(), reason="Requires virtual environment")
    def test_save_and_load_pipeline_state(self, tmp_path, mock_vector_store):
        """Test saving and loading pipeline state."""
        from RAGnificent.rag.pipeline import Pipeline
        
        pipeline = Pipeline(
            collection_name="test_persistence",
            data_dir=tmp_path
        )
        
        # Save pipeline configuration
        config_file = tmp_path / "pipeline_config.yaml"
        pipeline.save_config(config_file)
        
        assert config_file.exists()
        
        # Load pipeline from configuration
        loaded_pipeline = Pipeline(config=config_file)
        assert loaded_pipeline.collection_name == "test_persistence"

    def test_metadata_preservation(self, mock_scraper, mock_embedding_service, mock_vector_store):
        """Test that metadata is preserved throughout the pipeline."""
        from RAGnificent.utils.chunk_utils import ContentChunker
        
        url = "https://example.com/test"
        markdown = mock_scraper.scrape_website(url)
        
        chunker = ContentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.create_chunks_from_markdown(
            markdown,
            source_url=url,
            additional_metadata={"doc_type": "tutorial", "version": "1.0"}
        )
        
        # Verify metadata is preserved
        for chunk in chunks:
            assert chunk["metadata"]["source_url"] == url
            assert chunk["metadata"]["doc_type"] == "tutorial"
            assert chunk["metadata"]["version"] == "1.0"
            assert "chunk_index" in chunk["metadata"]