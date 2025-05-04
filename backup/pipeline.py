"""
RAG Pipeline module for RAGnificent.
Provides a complete end-to-end pipeline for building and using RAG systems.
Combines Rust-optimized scraping with efficient embeddings and vector search.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from RAGnificent.core.scraper import MarkdownScraper
from RAGnificent.utils.chunk_utils import ContentChunker, create_semantic_chunks
from RAGnificent.rag.embedding import get_embedding_service
from RAGnificent.rag.vector_store import get_vector_store
from RAGnificent.rag.search import get_search

# Import agent from v1 implementation for LLM integration
from v1_implementation.agent import query_with_context, summarize_documents

logger = logging.getLogger(__name__)

class RAGPipeline:
    """End-to-end RAG pipeline combining the best of both implementations."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        requests_per_second: float = 1.0,
        cache_enabled: bool = True,
        data_dir: Optional[str] = None
    ):
        """
        Initialize the RAG pipeline.

        Args:
            collection_name: Name of the vector collection
            embedding_model: Name of embedding model
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            requests_per_second: Maximum number of requests per second for scraping
            cache_enabled: Whether to enable request caching
            data_dir: Directory to save data files
        """
        # Initialize scraper with Rust optimization when available
        self.scraper = MarkdownScraper(
            requests_per_second=requests_per_second,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cache_enabled=cache_enabled
        )
        
        # Initialize chunker
        self.chunker = ContentChunker(chunk_size, chunk_overlap)
        
        # Initialize embedding service
        self.embedding_service = get_embedding_service(embedding_model)
        
        # Initialize vector store
        self.vector_store = get_vector_store(collection_name)
        
        # Initialize search
        self.search = get_search(collection_name, embedding_model)
        
        # Set data directory
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info(f"Initialized RAG pipeline with collection: {collection_name}")
        
    def extract_content(
        self, 
        url: str = None,
        sitemap_url: str = None,
        links_file: str = None,
        limit: Optional[int] = None,
        output_file: Optional[str] = None,
        output_format: str = "markdown"
    ) -> List[Dict[str, Any]]:
        """
        Extract content from URLs using Rust-optimized scraper.

        Args:
            url: Single URL to scrape
            sitemap_url: URL to sitemap for bulk scraping
            links_file: Path to file containing links to scrape
            limit: Maximum number of URLs to process
            output_file: Path to save raw output
            output_format: Format of output (markdown, json, or xml)

        Returns:
            List of document dictionaries
        """
        documents = []
        
        if url:
            logger.info(f"Extracting content from URL: {url}")
            try:
                # Use the Rust-optimized scraper when available
                html_content = self.scraper.scrape_website(url)
                content = self.scraper.convert_html(
                    html_content, url, output_format
                )
                
                document = {
                    "url": url,
                    "content": content,
                    "title": self._extract_title(content, url),
                    "format": output_format
                }
                documents.append(document)
                
            except Exception as e:
                logger.error(f"Error extracting content from {url}: {e}")
        
        elif sitemap_url:
            logger.info(f"Extracting content from sitemap: {sitemap_url}")
            try:
                # Get URLs from sitemap
                from RAGnificent.utils.sitemap_utils import SitemapParser
                parser = SitemapParser()
                urls = parser.parse_sitemap(sitemap_url, limit=limit)
                
                for url in urls:
                    try:
                        html_content = self.scraper.scrape_website(url)
                        content = self.scraper.convert_html(
                            html_content, url, output_format
                        )
                        
                        document = {
                            "url": url,
                            "content": content,
                            "title": self._extract_title(content, url),
                            "format": output_format
                        }
                        documents.append(document)
                        
                    except Exception as e:
                        logger.error(f"Error extracting content from {url}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing sitemap {sitemap_url}: {e}")
        
        elif links_file:
            logger.info(f"Extracting content from links in file: {links_file}")
            try:
                with open(links_file, 'r', encoding='utf-8') as f:
                    urls = [line.strip() for line in f if line.strip()]
                
                # Apply limit if specified
                if limit:
                    urls = urls[:limit]
                
                for url in urls:
                    try:
                        html_content = self.scraper.scrape_website(url)
                        content = self.scraper.convert_html(
                            html_content, url, output_format
                        )
                        
                        document = {
                            "url": url,
                            "content": content,
                            "title": self._extract_title(content, url),
                            "format": output_format
                        }
                        documents.append(document)
                        
                    except Exception as e:
                        logger.error(f"Error extracting content from {url}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error reading links file {links_file}: {e}")
        
        # Save raw documents if output file specified
        if output_file and documents:
            output_path = self.data_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(documents)} raw documents to {output_path}")
            
        return documents
    
    def chunk_documents(
        self, 
        documents: Union[List[Dict[str, Any]], str],
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk documents into smaller pieces for RAG.

        Args:
            documents: List of document dictionaries or path to documents file
            output_file: Path to save chunks

        Returns:
            List of chunk dictionaries
        """
        # Load documents from file if path provided
        if isinstance(documents, str):
            try:
                with open(documents, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
            except Exception as e:
                logger.error(f"Error loading documents from {documents}: {e}")
                return []
        
        # Ensure we have documents
        if not documents:
            logger.warning("No documents to chunk")
            return []
            
        logger.info(f"Chunking {len(documents)} documents")
        all_chunks = []
        
        for doc in documents:
            url = doc.get("url", "")
            content = doc.get("content", "")
            
            if not content:
                continue
                
            try:
                # Use our advanced chunker
                chunks = self.chunker.create_chunks_from_markdown(content, url)
                # Convert Chunk objects to dictionaries
                chunk_dicts = [
                    {
                        "id": chunk.id,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "source_url": chunk.source_url,
                        "created_at": chunk.created_at,
                        "chunk_type": chunk.chunk_type
                    }
                    for chunk in chunks
                ]
                all_chunks.extend(chunk_dicts)
                
            except Exception as e:
                logger.error(f"Error chunking document from {url}: {e}")
                continue
                
        # Save chunks if output file specified
        if output_file and all_chunks:
            output_path = self.data_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(all_chunks)} chunks to {output_path}")
            
        return all_chunks
    
    def embed_chunks(
        self, 
        chunks: Union[List[Dict[str, Any]], str],
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries or path to chunks file
            output_file: Path to save embedded chunks

        Returns:
            List of chunk dictionaries with embeddings
        """
        # Load chunks from file if path provided
        if isinstance(chunks, str):
            try:
                with open(chunks, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
            except Exception as e:
                logger.error(f"Error loading chunks from {chunks}: {e}")
                return []
        
        # Ensure we have chunks
        if not chunks:
            logger.warning("No chunks to embed")
            return []
            
        logger.info(f"Embedding {len(chunks)} chunks")
        
        try:
            # Generate embeddings in batch
            embedded_chunks = self.embedding_service.embed_chunks(chunks)
            
            # Save embedded chunks if output file specified
            if output_file and embedded_chunks:
                output_path = self.data_dir / output_file
                with open(output_path, 'w', encoding='utf-8') as f:
                    # Store chunks without embeddings to save space
                    lightweight_chunks = []
                    for chunk in embedded_chunks:
                        # Store info that we have embeddings but don't store the actual embeddings
                        chunk_copy = {k: v for k, v in chunk.items() if k != "embedding"}
                        chunk_copy["has_embedding"] = "embedding" in chunk
                        lightweight_chunks.append(chunk_copy)
                    
                    json.dump(lightweight_chunks, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(embedded_chunks)} embedded chunks info to {output_path}")
                
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            return chunks  # Return original chunks without embeddings
    
    def store_chunks(self, chunks: Union[List[Dict[str, Any]], str]) -> bool:
        """
        Store chunks in vector database.

        Args:
            chunks: List of chunk dictionaries with embeddings or path to embedded chunks file

        Returns:
            Success flag
        """
        # Load chunks from file if path provided
        if isinstance(chunks, str):
            try:
                with open(chunks, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    
                # If these are lightweight chunks without embeddings, we need to re-embed them
                if chunks and all("has_embedding" in chunk and "embedding" not in chunk for chunk in chunks):
                    logger.info("Re-embedding chunks from file")
                    chunks = self.embedding_service.embed_chunks(chunks)
                    
            except Exception as e:
                logger.error(f"Error loading embedded chunks from {chunks}: {e}")
                return False
        
        # Ensure we have chunks with embeddings
        valid_chunks = [chunk for chunk in chunks if "embedding" in chunk]
        if not valid_chunks:
            logger.warning("No chunks with embeddings to store")
            return False
            
        logger.info(f"Storing {len(valid_chunks)} chunks in vector database")
        
        # Store chunks in vector database
        return self.vector_store.store_chunks(valid_chunks)
    
    def search_documents(
        self, 
        query: str,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to query.

        Args:
            query: Query text
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            List of matching chunks with scores
        """
        logger.info(f"Searching for: {query}")
        return self.search.search(query, limit, threshold)
    
    def query_with_rag(
        self, 
        query: str,
        limit: int = 5,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Query with RAG-enhanced results using LLM integration.

        Args:
            query: Query text
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            Dictionary with query, response, and context information
        """
        logger.info(f"RAG query: {query}")
        
        # Get relevant documents
        results = self.search_documents(query, limit, threshold)
        
        if not results:
            logger.warning(f"No relevant documents found for query: {query}")
            return {
                "query": query,
                "response": "I couldn't find any relevant information to answer your query.",
                "context": [],
                "has_context": False
            }
        
        # Extract context from results
        context = []
        for result in results:
            payload = result.get("payload", {})
            content = payload.get("content", "")
            source_url = payload.get("source_url", "")
            
            context.append({
                "content": content,
                "url": source_url,
                "score": result.get("score", 0)
            })
        
        # Use agent from v1 implementation to generate response
        response = query_with_context(query, context)
        
        return {
            "query": query,
            "response": response,
            "context": context,
            "has_context": bool(context)
        }
    
    def run_pipeline(
        self,
        url: str = None,
        sitemap_url: str = None,
        links_file: str = None, 
        limit: Optional[int] = None,
        output_format: str = "markdown",
        run_extract: bool = True,
        run_chunk: bool = True,
        run_embed: bool = True,
        run_store: bool = True
    ) -> bool:
        """
        Run the complete RAG pipeline.

        Args:
            url: Single URL to process
            sitemap_url: URL to sitemap for bulk processing
            links_file: Path to file containing links to process
            limit: Maximum number of URLs to process
            output_format: Format of output (markdown, json, or xml)
            run_extract: Whether to run extraction step
            run_chunk: Whether to run chunking step
            run_embed: Whether to run embedding step
            run_store: Whether to run storage step

        Returns:
            Success flag
        """
        logger.info("Starting RAG pipeline")
        
        documents = []
        chunks = []
        embedded_chunks = []
        
        # Step 1: Extract content
        if run_extract:
            documents = self.extract_content(
                url=url,
                sitemap_url=sitemap_url,
                links_file=links_file,
                limit=limit,
                output_file="raw_documents.json",
                output_format=output_format
            )
            
            if not documents:
                logger.error("Extraction step failed - no documents extracted")
                return False
                
            logger.info(f"Extraction step completed: {len(documents)} documents")
        
        # Step 2: Chunk documents
        if run_chunk:
            if not documents and run_extract:
                logger.error("No documents to chunk")
                return False
            
            # If we didn't run extract, try to load documents from file
            if not documents:
                documents = self.data_dir / "raw_documents.json"
                
            chunks = self.chunk_documents(
                documents=documents,
                output_file="document_chunks.json"
            )
            
            if not chunks:
                logger.error("Chunking step failed - no chunks created")
                return False
                
            logger.info(f"Chunking step completed: {len(chunks)} chunks")
        
        # Step 3: Embed chunks
        if run_embed:
            if not chunks and run_chunk:
                logger.error("No chunks to embed")
                return False
            
            # If we didn't run chunk, try to load chunks from file
            if not chunks:
                chunks = self.data_dir / "document_chunks.json"
                
            embedded_chunks = self.embed_chunks(
                chunks=chunks,
                output_file="embedded_chunks.json"
            )
            
            if not embedded_chunks:
                logger.error("Embedding step failed - no embeddings created")
                return False
                
            logger.info(f"Embedding step completed: {len(embedded_chunks)} embedded chunks")
        
        # Step 4: Store chunks
        if run_store:
            if not embedded_chunks and run_embed:
                logger.error("No embedded chunks to store")
                return False
            
            # If we didn't run embed, try to load embedded chunks from file
            if not embedded_chunks:
                embedded_chunks = self.data_dir / "embedded_chunks.json"
                
            success = self.store_chunks(embedded_chunks)
            
            if not success:
                logger.error("Storage step failed")
                return False
                
            logger.info("Storage step completed")
        
        logger.info("RAG pipeline completed successfully")
        return True
    
    def _extract_title(self, content: str, url: str) -> str:
        """Extract title from content."""
        # For markdown content, extract first h1 heading
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        
        # Fallback to URL-based title
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        path = parsed_url.path.strip('/')
        if path:
            return path.split('/')[-1].replace('-', ' ').replace('_', ' ').title()
        return parsed_url.netloc
