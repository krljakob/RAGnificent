"""
RAG Pipeline module for RAGnificent.

Provides a complete end-to-end pipeline for Retrieval Augmented Generation,
combining Rust-optimized scraping with embedding, vector storage, and search.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from RAGnificent.core.config import ChunkingStrategy, get_config
from RAGnificent.core.scraper import MarkdownScraper
from RAGnificent.rag.embedding import get_embedding_service
from RAGnificent.rag.search import SearchResult, get_search
from RAGnificent.rag.vector_store import get_vector_store
from RAGnificent.utils.chunk_utils import ContentChunker

# Optional LLM integration for RAG
try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end RAG pipeline that combines scraping, chunking, embedding, and search."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model_type: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        requests_per_second: Optional[float] = None,
        cache_enabled: Optional[bool] = None,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            collection_name: Name of the vector collection
            embedding_model_type: Type of embedding model to use
            embedding_model_name: Name of embedding model
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            requests_per_second: Maximum number of requests per second for scraping
            cache_enabled: Whether to enable request caching
            data_dir: Directory to save data files
        """
        # Load configuration
        self.config = get_config()

        # Set up data directory
        self.data_dir = Path(data_dir) if data_dir else Path(self.config.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize scraper with Rust optimization when available
        self.scraper = MarkdownScraper(
            requests_per_second=requests_per_second or self.config.scraper.rate_limit,
            chunk_size=chunk_size or self.config.chunking.chunk_size,
            chunk_overlap=chunk_overlap or self.config.chunking.chunk_overlap,
            cache_enabled=(
                cache_enabled
                if cache_enabled is not None
                else self.config.scraper.cache_enabled
            ),
        )

        # Initialize chunker
        self.chunker = ContentChunker(
            chunk_size or self.config.chunking.chunk_size,
            chunk_overlap or self.config.chunking.chunk_overlap,
        )

        # Initialize embedding service
        self.embedding_service = get_embedding_service(
            embedding_model_type, embedding_model_name
        )

        # Initialize vector store
        self.vector_store = get_vector_store(collection_name)

        # Initialize search
        self.search = get_search(
            collection_name, embedding_model_type, embedding_model_name
        )

        # Set collection name
        self.collection_name = collection_name or self.config.qdrant.collection

        logger.info(f"Initialized RAG pipeline with collection: {self.collection_name}")

    def extract_content(
        self,
        url: Optional[str] = None,
        urls: Optional[List[str]] = None,
        sitemap_url: Optional[str] = None,
        links_file: Optional[str] = None,
        limit: Optional[int] = None,
        output_file: Optional[str] = None,
        output_format: str = "markdown",
        skip_cache: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract content from URLs using Rust-optimized scraper.

        Args:
            url: Single URL to scrape
            urls: List of URLs to scrape
            sitemap_url: URL to sitemap for bulk scraping
            links_file: Path to file containing links to scrape
            limit: Maximum number of URLs to process
            output_file: Path to save raw output
            output_format: Format of output (markdown, json, or xml)
            skip_cache: Whether to bypass the request cache

        Returns:
            List of document dictionaries
        """
        documents = []

        # Process single URL
        if url:
            logger.info(f"Extracting content from URL: {url}")
            try:
                # Use the Rust-optimized scraper when available
                html_content = self.scraper.scrape_website(url, skip_cache=skip_cache)
                content = self.scraper.convert_html(html_content, url, output_format)

                document = {
                    "url": url,
                    "content": content,
                    "title": self._extract_title(content, url),
                    "format": output_format,
                    "timestamp": datetime.now().isoformat(),
                }
                documents.append(document)

            except Exception as e:
                logger.error(f"Error extracting content from {url}: {e}")

        # Process list of URLs
        elif urls:
            logger.info(f"Extracting content from {len(urls)} URLs")

            # Apply limit if specified
            if limit and len(urls) > limit:
                urls = urls[:limit]

            for url in urls:
                try:
                    html_content = self.scraper.scrape_website(
                        url, skip_cache=skip_cache
                    )
                    content = self.scraper.convert_html(
                        html_content, url, output_format
                    )

                    document = {
                        "url": url,
                        "content": content,
                        "title": self._extract_title(content, url),
                        "format": output_format,
                        "timestamp": datetime.now().isoformat(),
                    }
                    documents.append(document)

                except Exception as e:
                    logger.error(f"Error extracting content from {url}: {e}")
                    continue

        # Process sitemap URL
        elif sitemap_url:
            logger.info(f"Extracting content from sitemap: {sitemap_url}")
            try:
                # Get URLs from sitemap
                from RAGnificent.utils.sitemap_utils import SitemapParser

                parser = SitemapParser()
                sitemap_urls = parser.parse_sitemap(sitemap_url, limit=limit)

                for url in sitemap_urls:
                    try:
                        html_content = self.scraper.scrape_website(
                            url, skip_cache=skip_cache
                        )
                        content = self.scraper.convert_html(
                            html_content, url, output_format
                        )

                        document = {
                            "url": url,
                            "content": content,
                            "title": self._extract_title(content, url),
                            "format": output_format,
                            "timestamp": datetime.now().isoformat(),
                        }
                        documents.append(document)

                    except Exception as e:
                        logger.error(f"Error extracting content from {url}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error processing sitemap {sitemap_url}: {e}")

        # Process links file
        elif links_file:
            logger.info(f"Extracting content from links in file: {links_file}")
            try:
                with open(links_file, "r", encoding="utf-8") as f:
                    file_urls = [line.strip() for line in f if line.strip()]

                # Apply limit if specified
                if limit:
                    file_urls = file_urls[:limit]

                for url in file_urls:
                    try:
                        html_content = self.scraper.scrape_website(
                            url, skip_cache=skip_cache
                        )
                        content = self.scraper.convert_html(
                            html_content, url, output_format
                        )

                        document = {
                            "url": url,
                            "content": content,
                            "title": self._extract_title(content, url),
                            "format": output_format,
                            "timestamp": datetime.now().isoformat(),
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
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(documents, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(documents)} raw documents to {output_path}")

        return documents

    def chunk_documents(
        self,
        documents: Union[List[Dict[str, Any]], str],
        output_file: Optional[str] = None,
        strategy: Optional[ChunkingStrategy] = None,
    ) -> List[Dict[str, Any]]:
        """
        Chunk documents into smaller pieces for RAG.

        Args:
            documents: List of document dictionaries or path to documents file
            output_file: Path to save chunks
            strategy: Chunking strategy to use

        Returns:
            List of chunk dictionaries
        """
        # Load documents from file if path provided
        if isinstance(documents, str):
            try:
                with open(documents, "r", encoding="utf-8") as f:
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

        # Get chunking strategy from config if not specified
        chunking_strategy = strategy or self.config.chunking.strategy

        for doc in documents:
            url = doc.get("url", "")
            content = doc.get("content", "")
            doc_title = doc.get("title", "Untitled Document")

            if not content:
                continue

            try:
                if chunking_strategy == ChunkingStrategy.SEMANTIC:
                    # Use our advanced semantic chunker
                    chunks = self.chunker.create_chunks_from_markdown(content, url)
                    # Convert Chunk objects to dictionaries
                    chunk_dicts = [
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "metadata": {
                                **chunk.metadata,
                                "title": doc_title,
                                "document_url": url,
                            },
                            "source_url": chunk.source_url,
                            "created_at": chunk.created_at,
                            "chunk_type": chunk.chunk_type,
                        }
                        for chunk in chunks
                    ]
                    all_chunks.extend(chunk_dicts)

                elif chunking_strategy == ChunkingStrategy.SLIDING_WINDOW:
                    # Use sliding window chunking (simpler)
                    from RAGnificent.utils.chunk_utils import chunk_text

                    chunk_size = self.config.chunking.chunk_size
                    chunk_overlap = self.config.chunking.chunk_overlap

                    text_chunks = chunk_text(
                        content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )

                    for i, chunk_text in enumerate(text_chunks):
                        chunk_dict = {
                            "id": f"{url}_{i}",
                            "content": chunk_text,
                            "metadata": {
                                "title": doc_title,
                                "document_url": url,
                                "chunk_index": i,
                                "chunk_count": len(text_chunks),
                            },
                            "source_url": url,
                            "created_at": datetime.now().isoformat(),
                            "chunk_type": "sliding_window",
                        }
                        all_chunks.append(chunk_dict)

                elif chunking_strategy == ChunkingStrategy.RECURSIVE:
                    # Use recursive chunking (for hierarchical documents)
                    from RAGnificent.utils.chunk_utils import recursive_chunk_text

                    chunk_size = self.config.chunking.chunk_size
                    chunk_overlap = self.config.chunking.chunk_overlap

                    recursive_chunks = recursive_chunk_text(
                        content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )

                    for i, chunk_text in enumerate(recursive_chunks):
                        chunk_dict = {
                            "id": f"{url}_{i}",
                            "content": chunk_text,
                            "metadata": {
                                "title": doc_title,
                                "document_url": url,
                                "chunk_index": i,
                                "chunk_count": len(recursive_chunks),
                            },
                            "source_url": url,
                            "created_at": datetime.now().isoformat(),
                            "chunk_type": "recursive",
                        }
                        all_chunks.append(chunk_dict)

            except Exception as e:
                logger.error(f"Error chunking document from {url}: {e}")
                continue

        # Save chunks if output file specified
        if output_file and all_chunks:
            output_path = self.data_dir / output_file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(all_chunks)} chunks to {output_path}")

        return all_chunks

    def embed_chunks(
        self,
        chunks: Union[List[Dict[str, Any]], str],
        output_file: Optional[str] = None,
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
                with open(chunks, "r", encoding="utf-8") as f:
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
                with open(output_path, "w", encoding="utf-8") as f:
                    # Store chunks without embeddings to save space
                    lightweight_chunks = []
                    for chunk in embedded_chunks:
                        # Store info that we have embeddings but don't store the actual embeddings
                        chunk_copy = {
                            k: v for k, v in chunk.items() if k != "embedding"
                        }
                        chunk_copy["has_embedding"] = "embedding" in chunk
                        lightweight_chunks.append(chunk_copy)

                    json.dump(lightweight_chunks, f, indent=2, ensure_ascii=False)
                logger.info(
                    f"Saved {len(embedded_chunks)} embedded chunks info to {output_path}"
                )

            return embedded_chunks

        except Exception as e:
            logger.error(f"Error embedding chunks: {e}")
            return chunks  # Return original chunks without embeddings

    def store_chunks(
        self,
        chunks: Union[List[Dict[str, Any]], str],
        embedding_field: str = "embedding",
        id_field: str = "id",
    ) -> bool:
        """
        Store chunks in vector database.

        Args:
            chunks: List of chunk dictionaries with embeddings or path to embedded chunks file
            embedding_field: Field name containing the embedding
            id_field: Field name containing the document ID

        Returns:
            Success flag
        """
        # Load chunks from file if path provided
        if isinstance(chunks, str):
            try:
                with open(chunks, "r", encoding="utf-8") as f:
                    chunks = json.load(f)

                # If these are lightweight chunks without embeddings, we need to re-embed them
                if chunks and all(
                    "has_embedding" in chunk and "embedding" not in chunk
                    for chunk in chunks
                ):
                    logger.info("Re-embedding chunks from file")
                    chunks = self.embedding_service.embed_chunks(chunks)

            except Exception as e:
                logger.error(f"Error loading embedded chunks from {chunks}: {e}")
                return False

        # Ensure we have chunks with embeddings
        valid_chunks = [chunk for chunk in chunks if embedding_field in chunk]
        if not valid_chunks:
            logger.warning("No chunks with embeddings to store")
            return False

        logger.info(f"Storing {len(valid_chunks)} chunks in vector database")

        # Store chunks in vector database
        return self.vector_store.store_documents(
            valid_chunks, embedding_field=embedding_field, id_field=id_field
        )

    def search_documents(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
        filter_conditions: Optional[Dict[str, Any]] = None,
        rerank: bool = False,
        as_dict: bool = True,
    ) -> Union[List[SearchResult], List[Dict[str, Any]]]:
        """
        Search for documents similar to query.

        Args:
            query: Query text
            limit: Maximum number of results
            threshold: Similarity threshold
            filter_conditions: Additional filter conditions
            rerank: Whether to rerank results
            as_dict: Whether to return results as dictionaries

        Returns:
            List of matching chunks with scores
        """
        logger.info(f"Searching for: {query}")
        results = self.search.search(query, limit, threshold, filter_conditions, rerank)

        return [r.to_dict() for r in results] if as_dict else results

    def query_with_context(
        self,
        query: str,
        limit: int = 5,
        threshold: float = 0.7,
        model: str = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response to the query using retrieved context.

        Args:
            query: The query to answer
            limit: Maximum number of context chunks
            threshold: Similarity threshold
            model: LLM model to use
            temperature: Temperature for generation
            system_prompt: Custom system prompt

        Returns:
            Dictionary with query, response, and context
        """
        if openai is None:
            logger.error("OpenAI package not installed, cannot generate response")
            return {
                "query": query,
                "response": "Error: OpenAI package not installed for LLM integration",
                "context": [],
                "has_context": False,
            }

        # Get relevant documents
        results = self.search_documents(query, limit, threshold, as_dict=True)

        if not results:
            logger.warning(f"No relevant documents found for query: {query}")
            return {
                "query": query,
                "response": "I couldn't find any relevant information to answer your query.",
                "context": [],
                "has_context": False,
            }

        # Use default model from config if not specified
        if not model:
            model = self.config.openai.completion_model

        # Format context for prompt
        context_str = "\n\n".join(
            f"DOCUMENT {i+1}:\n{result['content']}" for i, result in enumerate(results)
        )

        # Create default system prompt if none provided
        if system_prompt is None:
            system_prompt = """You are a helpful assistant. Answer the question based on the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer that."
Always cite your sources by referencing the document numbers.
"""

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {query}",
            },
        ]

        try:
            # Generate response
            completion = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=self.config.openai.max_tokens,
                timeout=self.config.openai.request_timeout,
            )

            response = completion.choices[0].message.content

            return {
                "query": query,
                "response": response,
                "context": results,
                "has_context": True,
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "query": query,
                "response": f"Error generating response: {str(e)}",
                "context": results,
                "has_context": True,
            }

    def run_pipeline(
        self,
        url: Optional[str] = None,
        urls: Optional[List[str]] = None,
        sitemap_url: Optional[str] = None,
        links_file: Optional[str] = None,
        limit: Optional[int] = None,
        output_format: str = "markdown",
        run_extract: bool = True,
        run_chunk: bool = True,
        run_embed: bool = True,
        run_store: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.

        Args:
            url: Single URL to process
            urls: List of URLs to process
            sitemap_url: URL to sitemap for bulk processing
            links_file: Path to file containing links to process
            limit: Maximum number of URLs to process
            output_format: Format of output (markdown, json, or xml)
            run_extract: Whether to run extraction step
            run_chunk: Whether to run chunking step
            run_embed: Whether to run embedding step
            run_store: Whether to run storage step

        Returns:
            Dictionary with pipeline status and document counts
        """
        logger.info("Starting RAG pipeline")
        result = {
            "success": True,
            "steps": {},
            "timestamp": datetime.now().isoformat(),
            "document_counts": {},
        }

        documents = []
        chunks = []
        embedded_chunks = []

        # Step 1: Extract content
        if run_extract:
            try:
                documents = self.extract_content(
                    url=url,
                    urls=urls,
                    sitemap_url=sitemap_url,
                    links_file=links_file,
                    limit=limit,
                    output_file="raw_documents.json",
                    output_format=output_format,
                )

                if not documents:
                    logger.error("Extraction step failed - no documents extracted")
                    result["steps"]["extract"] = False
                    result["success"] = False
                    return result

                logger.info(f"Extraction step completed: {len(documents)} documents")
                result["steps"]["extract"] = True
                result["document_counts"]["documents"] = len(documents)
            except Exception as e:
                logger.error(f"Extraction step failed: {e}")
                result["steps"]["extract"] = False
                result["success"] = False
                return result

        # Step 2: Chunk documents
        if run_chunk:
            try:
                if not documents and run_extract:
                    logger.error("No documents to chunk")
                    result["steps"]["chunk"] = False
                    result["success"] = False
                    return result

                # If we didn't run extract, try to load documents from file
                if not documents:
                    documents = str(self.data_dir / "raw_documents.json")

                chunks = self.chunk_documents(
                    documents=documents, output_file="document_chunks.json"
                )

                if not chunks:
                    logger.error("Chunking step failed - no chunks created")
                    result["steps"]["chunk"] = False
                    result["success"] = False
                    return result

                logger.info(f"Chunking step completed: {len(chunks)} chunks")
                result["steps"]["chunk"] = True
                result["document_counts"]["chunks"] = len(chunks)
            except Exception as e:
                logger.error(f"Chunking step failed: {e}")
                result["steps"]["chunk"] = False
                result["success"] = False
                return result

        # Step 3: Embed chunks
        if run_embed:
            try:
                if not chunks and run_chunk:
                    logger.error("No chunks to embed")
                    result["steps"]["embed"] = False
                    result["success"] = False
                    return result

                # If we didn't run chunk, try to load chunks from file
                if not chunks:
                    chunks = str(self.data_dir / "document_chunks.json")

                embedded_chunks = self.embed_chunks(
                    chunks=chunks, output_file="embedded_chunks.json"
                )

                if not embedded_chunks:
                    logger.error("Embedding step failed - no embeddings created")
                    result["steps"]["embed"] = False
                    result["success"] = False
                    return result

                logger.info(
                    f"Embedding step completed: {len(embedded_chunks)} embedded chunks"
                )
                result["steps"]["embed"] = True
                result["document_counts"]["embedded_chunks"] = len(embedded_chunks)
            except Exception as e:
                logger.error(f"Embedding step failed: {e}")
                result["steps"]["embed"] = False
                result["success"] = False
                return result

        # Step 4: Store chunks
        if run_store:
            try:
                if not embedded_chunks and run_embed:
                    logger.error("No embedded chunks to store")
                    result["steps"]["store"] = False
                    result["success"] = False
                    return result

                # If we didn't run embed, try to load embedded chunks from file
                if not embedded_chunks:
                    embedded_chunks = str(self.data_dir / "embedded_chunks.json")

                success = self.store_chunks(embedded_chunks)

                if not success:
                    logger.error("Storage step failed")
                    result["steps"]["store"] = False
                    result["success"] = False
                    return result

                logger.info("Storage step completed")
                result["steps"]["store"] = True

                # Count documents in vector store
                doc_count = self.vector_store.count_documents()
                result["document_counts"]["stored_vectors"] = doc_count
            except Exception as e:
                logger.error(f"Storage step failed: {e}")
                result["steps"]["store"] = False
                result["success"] = False
                return result

        logger.info("RAG pipeline completed successfully")
        return result

    def _extract_title(self, content: str, url: str) -> str:
        """Extract title from content."""
        # For markdown content, extract first h1 heading
        lines = content.split("\n")
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()

        # Fallback to URL-based title
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        path = parsed_url.path.strip("/")
        if path:
            return path.split("/")[-1].replace("-", " ").replace("_", " ").title()
        return parsed_url.netloc


# Singleton instance for easy access
_default_pipeline = None


def get_pipeline(
    collection_name: Optional[str] = None,
    embedding_model_type: Optional[str] = None,
    embedding_model_name: Optional[str] = None,
) -> Pipeline:
    """
    Get or create the default pipeline.

    Args:
        collection_name: Name of the vector collection
        embedding_model_type: Type of embedding model to use
        embedding_model_name: Name of embedding model

    Returns:
        The pipeline instance
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = Pipeline(
            collection_name=collection_name,
            embedding_model_type=embedding_model_type,
            embedding_model_name=embedding_model_name,
        )
    return _default_pipeline
