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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

from ..core.config import ChunkingStrategy, EmbeddingModelType, get_config
from ..core.scraper import MarkdownScraper
from ..utils.chunk_utils import ContentChunker
from .embedding import get_embedding_service
from .search import SearchResult, get_search
from .vector_store import get_vector_store

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
        config: Optional[Union[Dict[str, Any], str, Path]] = None,
        collection_name: Optional[str] = None,
        embedding_model_type: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        requests_per_second: Optional[float] = None,
        cache_enabled: Optional[bool] = None,
        data_dir: Optional[Union[str, Path]] = None,
        continue_on_error: bool = False,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            config: Pipeline configuration (dict, YAML file path, or config dict)
            collection_name: Name of the vector collection
            embedding_model_type: Type of embedding model to use
            embedding_model_name: Name of embedding model
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            requests_per_second: Maximum number of requests per second for scraping
            cache_enabled: Whether to enable request caching
            data_dir: Directory to save data files
            continue_on_error: Whether to continue pipeline execution on errors
        """
        # Load pipeline configuration
        self.pipeline_config = self._load_pipeline_config(config)
        self.continue_on_error = continue_on_error

        # Load system configuration
        self.config = get_config()

        # Resolve configuration parameters (CLI args override pipeline config override system config)
        self.collection_name = (
            collection_name
            or self.pipeline_config.get("collection_name")
            or self.config.qdrant.collection
        )

        resolved_chunk_size = (
            chunk_size
            or self.pipeline_config.get("chunk_size")
            or self.config.chunking.chunk_size
        )

        resolved_chunk_overlap = (
            chunk_overlap
            or self.pipeline_config.get("chunk_overlap")
            or self.config.chunking.chunk_overlap
        )

        resolved_requests_per_second = (
            requests_per_second
            or self.pipeline_config.get("requests_per_second")
            or self.config.scraper.rate_limit
        )

        resolved_cache_enabled = (
            cache_enabled
            if cache_enabled is not None
            else self.pipeline_config.get(
                "cache_enabled", self.config.scraper.cache_enabled
            )
        )

        resolved_embedding_model_type = (
            embedding_model_type or self.pipeline_config.get("embedding_model_type")
        )

        resolved_embedding_model_name = (
            embedding_model_name or self.pipeline_config.get("embedding_model_name")
        )

        # Set up data directory
        pipeline_data_dir = self.pipeline_config.get(
            "data_dir"
        ) or self.pipeline_config.get("output_dir")
        resolved_data_dir = Path(data_dir or pipeline_data_dir or self.config.data_dir).resolve(strict=True)
        
        # Define safe root directory for data operations
        safe_root_dir = Path(self.config.data_dir_root).resolve(strict=True)
        
        # Ensure the resolved data directory is within the safe root directory
        if not resolved_data_dir.is_relative_to(safe_root_dir):
            raise ValueError(
                f"Data directory {resolved_data_dir} is outside the allowed root directory {safe_root_dir}"
            )
        
        # Check for symbolic links pointing outside the safe root directory
        if not resolved_data_dir.samefile(resolved_data_dir):
            raise ValueError(
                f"Data directory {resolved_data_dir} contains symbolic links pointing outside the allowed root directory {safe_root_dir}"
            )
        
        self.data_dir = resolved_data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize scraper with enhanced throttling and parallel processing
        self.scraper = MarkdownScraper(
            requests_per_second=resolved_requests_per_second,
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_chunk_overlap,
            cache_enabled=resolved_cache_enabled,
            domain_specific_limits=(
                self.config.scraper.domain_rate_limits
                if hasattr(self.config.scraper, "domain_rate_limits")
                else None
            ),
            max_workers=(
                self.config.scraper.max_workers
                if hasattr(self.config.scraper, "max_workers")
                else 10
            ),
            adaptive_throttling=(
                self.config.scraper.adaptive_throttling
                if hasattr(self.config.scraper, "adaptive_throttling")
                else True
            ),
        )

        # Initialize chunker
        self.chunker = ContentChunker(
            resolved_chunk_size,
            resolved_chunk_overlap,
        )

        # Initialize embedding service
        from ..core.config import EmbeddingModelType

        # Convert string to EmbeddingModelType enum if provided
        model_type_enum = None
        if resolved_embedding_model_type:
            if isinstance(resolved_embedding_model_type, str):
                model_type_enum = EmbeddingModelType(resolved_embedding_model_type)
            else:
                model_type_enum = resolved_embedding_model_type

        self.embedding_service = get_embedding_service(
            model_type_enum, resolved_embedding_model_name
        )

        # Initialize vector store
        self.vector_store = get_vector_store(self.collection_name)

        # Initialize search
        self.search = get_search(
            self.collection_name,
            resolved_embedding_model_type,
            resolved_embedding_model_name,
        )

        logger.info(f"Initialized RAG pipeline with collection: {self.collection_name}")

    def _load_pipeline_config(
        self, config: Optional[Union[Dict[str, Any], str, Path]]
    ) -> Dict[str, Any]:
        """Load pipeline configuration from various sources."""
        if config is None:
            return {}

        if isinstance(config, dict):
            return config

        # Load from file path
        # Normalize and resolve the user-provided path
        config_path = os.path.normpath(os.path.realpath(config))
        
        # Define safe root directories for configuration files
        project_root = Path(__file__).resolve().parent.parent.parent
        safe_roots = [
            os.path.realpath(project_root / "config"),
            os.path.realpath(project_root / "examples"),
            os.path.realpath(Path.cwd()),  # Current working directory
        ]
        
        # Check if the file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Pipeline configuration file not found: {config_path}"
            )
        
        # Ensure the config path is within allowed directories
        if not any(
            os.path.commonpath([config_path, safe_root]) == safe_root
            for safe_root in safe_roots
        ):
            raise ValueError(
                f"Access to configuration file outside allowed directories: {config_path}. "
                f"Allowed directories: {[str(r) for r in safe_roots]}"
            )

        with open(config_path, "r") as f:
            if config_path.suffix.lower() in {".yml", ".yaml"}:
                return yaml.safe_load(f) or {}
            if config_path.suffix.lower() == ".json":
                return json.load(f) or {}
            raise ValueError(
                f"Unsupported configuration file format: {config_path.suffix}"
            )

    def execute(self):
        """
        Execute the pipeline steps defined in the configuration.

        Yields step results for progress tracking.
        """
        if not self.pipeline_config.get("steps"):
            logger.warning("No pipeline steps defined")
            return

        for i, step in enumerate(self.pipeline_config["steps"]):
            step_name = step.get("name", f"Step {i+1}")
            step_type = step.get("type")
            step_config = step.get("config", {})

            logger.info(f"Executing step: {step_name} (type: {step_type})")

            try:
                if step_type == "scrape":
                    result = self._execute_scrape_step(step_config)
                elif step_type == "embed":
                    result = self._execute_embed_step(step_config)
                elif step_type == "index":
                    result = self._execute_index_step(step_config)
                elif step_type == "search":
                    result = self._execute_search_step(step_config)
                else:
                    raise ValueError(f"Unknown step type: {step_type}")

                yield {
                    "step_name": step_name,
                    "step_type": step_type,
                    "status": "success",
                    "result": result,
                }

            except Exception as e:
                logger.error(f"Error in step '{step_name}': {e}")
                yield {
                    "step_name": step_name,
                    "step_type": step_type,
                    "status": "error",
                    "error": str(e),
                }

                if not self.continue_on_error:
                    break

    def _execute_scrape_step(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a scraping step."""
        urls = config.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]

        output_format = config.get("format", "markdown")
        save_to_disk = config.get("save_to_disk", True)

        results = []
        for url in urls:
            try:
                if save_to_disk:
                    # Use existing scrape_and_index method
                    result = self.scrape_and_index([url])
                    results.append(
                        {
                            "url": url,
                            "status": "success",
                            "documents_created": len(
                                result.get("processed_documents", [])
                            ),
                        }
                    )
                else:
                    # Just scrape without saving
                    content = self.scraper.scrape_website(url)
                    results.append(
                        {
                            "url": url,
                            "status": "success",
                            "content_length": len(content),
                        }
                    )
            except Exception as e:
                results.append({"url": url, "status": "error", "error": str(e)})

        return {"scraped_urls": results}

    def _execute_embed_step(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an embedding step."""
        # For now, this is handled as part of the index step
        logger.info("Embedding step - handled during indexing")
        return {"status": "embeddings_handled_during_indexing"}

    def _execute_index_step(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an indexing step."""
        input_dir = config.get("input_dir", self.data_dir)
        input_dir_path = Path(input_dir).resolve()
        data_dir_path = Path(self.data_dir).resolve()
        
        # Ensure the input_dir is within the allowed data directory
        try:
            input_dir_path.relative_to(data_dir_path)
        except ValueError:
            raise ValueError(
                f"Input directory {input_dir_path} is outside the allowed data directory {data_dir_path}"
            )
        
        input_dir = str(input_dir_path)

        # Find all markdown files in input directory
        md_files = list(Path(input_dir).glob("*.md"))

        indexed_count = 0
        for md_file in md_files:
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Create chunks
                chunks = self.chunker.create_chunks_from_markdown(
                    content, source_url=str(md_file)
                )

                # Generate embeddings and store
                for chunk in chunks:
                    embedding = self.embedding_service.generate_embeddings(
                        [chunk["content"]]
                    )[0]
                    self.vector_store.store_documents(
                        [
                            {
                                "content": chunk["content"],
                                "metadata": chunk["metadata"],
                                "embedding": embedding,
                            }
                        ]
                    )

                indexed_count += len(chunks)

            except Exception as e:
                logger.error(f"Error indexing {md_file}: {e}")

        return {"indexed_documents": indexed_count}

    def _execute_search_step(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a search step."""
        query = config.get("query")
        if not query:
            raise ValueError("Search step requires 'query' parameter")

        top_k = config.get("top_k", 5)
        threshold = config.get("threshold", 0.7)

        results = self.search.search(query, top_k=top_k, threshold=threshold)

        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "content": (
                        f"{r.content[:200]}..." if len(r.content) > 200 else r.content
                    ),
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ],
        }

    def _process_single_url(
        self, url: str, output_format: str, skip_cache: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single URL and return a document dictionary.

        Args:
            url: URL to scrape
            output_format: Format of output (markdown, json, or xml)
            skip_cache: Whether to bypass the request cache

        Returns:
            Document dictionary or None if extraction failed
        """
        try:
            from ragnificent_rs import OutputFormat

            # Convert string to OutputFormat enum if it's not already an enum
            output_format_enum = output_format
            if isinstance(output_format, str):
                output_format_enum = OutputFormat(output_format)

            html_content = self.scraper.scrape_website(url, skip_cache=skip_cache)
            content = self.scraper.convert_html(html_content, url, output_format_enum)

            return {
                "url": url,
                "content": content,
                "title": self._extract_title(content, url),
                "format": output_format,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None

    def _process_url_list(
        self,
        urls: List[str],
        limit: Optional[int],
        output_format: str,
        skip_cache: bool,
    ) -> List[Dict[str, Any]]:
        """
        Process a list of URLs and return document dictionaries.

        Args:
            urls: List of URLs to scrape
            limit: Maximum number of URLs to process
            output_format: Format of output (markdown, json, or xml)
            skip_cache: Whether to bypass the request cache

        Returns:
            List of document dictionaries
        """
        documents = []

        # Apply limit if specified
        if limit and len(urls) > limit:
            urls = urls[:limit]

        for url in urls:
            if document := self._process_single_url(url, output_format, skip_cache):
                documents.append(document)

        return documents

    def _get_urls_from_sitemap(
        self, sitemap_url: str, limit: Optional[int]
    ) -> List[str]:
        """
        Extract URLs from a sitemap.

        Args:
            sitemap_url: URL to sitemap
            limit: Maximum number of URLs to retrieve

        Returns:
            List of URLs from the sitemap
        """
        try:
            from utils.sitemap_utils import SitemapParser

            parser = SitemapParser()
            sitemap_urls = parser.parse_sitemap(sitemap_url)

            # Extract URL strings from SitemapURL objects
            url_strings = [url.loc for url in sitemap_urls]

            # Then apply limit if specified
            if limit is not None and limit > 0 and url_strings:
                url_strings = url_strings[:limit]

            return url_strings
        except Exception as e:
            logger.error(f"Error processing sitemap {sitemap_url}: {e}")
            return []

    def _get_urls_from_file(self, links_file: str, limit: Optional[int]) -> List[str]:
        """
        Read URLs from a file.

        Args:
            links_file: Path to file containing links
            limit: Maximum number of URLs to retrieve

        Returns:
            List of URLs from the file
        """
        try:
            with open(links_file, "r", encoding="utf-8") as f:
                file_urls = [line.strip() for line in f if line.strip()]

            # Apply limit if specified
            if limit:
                file_urls = file_urls[:limit]

            return file_urls
        except Exception as e:
            logger.error(f"Error reading links file {links_file}: {e}")
            return []

    def _save_documents(
        self, documents: List[Dict[str, Any]], output_file: str
    ) -> None:
        """
        Save documents to a JSON file.

        Args:
            documents: List of document dictionaries to save
            output_file: Path to save documents to
        """
        if not documents:
            return

        output_path = self.data_dir / output_file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} raw documents to {output_path}")

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
        from ..core.security import redact_sensitive_data, secure_file_path
        from ..core.validators import (
            validate_file_path,
            validate_output_format,
            validate_url,
        )

        documents = []

        if not validate_output_format(output_format):
            logger.error(f"Invalid output format: {output_format}")
            return documents

        # Validate limit if provided
        if limit is not None and limit <= 0:
            logger.warning(f"Invalid limit value: {limit}, using default")
            limit = None

        # Process single URL
        if url:
            if not validate_url(url):
                logger.error(f"Invalid URL format: {redact_sensitive_data(url)}")
                return documents

            logger.info(f"Extracting content from URL: {redact_sensitive_data(url)}")
            if document := self._process_single_url(url, output_format, skip_cache):
                documents.append(document)

        elif urls:
            valid_urls = []
            for u in urls:
                if validate_url(u):
                    valid_urls.append(u)
                else:
                    logger.warning(f"Skipping invalid URL: {redact_sensitive_data(u)}")

            if not valid_urls:
                logger.error("No valid URLs found in the provided list")
                return documents

            logger.info(f"Extracting content from {len(valid_urls)} URLs")
            documents = self._process_url_list(
                valid_urls, limit, output_format, skip_cache
            )

        elif sitemap_url:
            if not validate_url(sitemap_url):
                logger.error(
                    f"Invalid sitemap URL format: {redact_sensitive_data(sitemap_url)}"
                )
                return documents

            logger.info(
                f"Extracting content from sitemap: {redact_sensitive_data(sitemap_url)}"
            )
            if sitemap_urls := self._get_urls_from_sitemap(sitemap_url, limit):
                documents = self._process_url_list(
                    sitemap_urls, None, output_format, skip_cache
                )

        elif links_file:
            if not validate_file_path(links_file, ["txt", "csv"]):
                logger.error(
                    f"Invalid links file path: {redact_sensitive_data(links_file)}"
                )
                return documents

            secure_path = secure_file_path(str(self.data_dir), links_file)

            logger.info(
                f"Extracting content from links in file: {redact_sensitive_data(secure_path)}"
            )
            if file_urls := self._get_urls_from_file(secure_path, limit):
                documents = self._process_url_list(
                    file_urls, None, output_format, skip_cache
                )

        # Save raw documents if output file specified
        if output_file:
            if not validate_file_path(output_file, ["json", "jsonl"]):
                logger.error(
                    f"Invalid output file path: {redact_sensitive_data(output_file)}"
                )
            else:
                secure_output_path = secure_file_path(str(self.data_dir), output_file)
                self._save_documents(documents, secure_output_path)

        return documents

    def _load_documents_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from a JSON file.

        Args:
            file_path: Path to the documents file

        Returns:
            List of document dictionaries
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading documents from {file_path}: {e}")
            return []

    def _create_semantic_chunks(
        self, content: str, url: str, doc_title: str
    ) -> List[Dict[str, Any]]:
        """
        Create semantic chunks from document content.

        Args:
            content: Document content to chunk
            url: Source URL
            doc_title: Document title

        Returns:
            List of chunk dictionaries
        """
        # Use our advanced semantic chunker
        chunks = self.chunker.create_chunks_from_markdown(content, url)

        # Convert Chunk objects to dictionaries
        return [
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

    def _create_sliding_window_chunks(
        self, content: str, url: str, doc_title: str
    ) -> List[Dict[str, Any]]:
        """
        Create sliding window chunks from document content.

        Args:
            content: Document content to chunk
            url: Source URL
            doc_title: Document title

        Returns:
            List of chunk dictionaries
        """
        from utils.chunk_utils import chunk_text

        chunk_size = self.config.chunking.chunk_size
        chunk_overlap = self.config.chunking.chunk_overlap

        text_chunks = chunk_text(
            content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        return [
            {
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
            for i, chunk_text in enumerate(text_chunks)
        ]

    def _create_recursive_chunks(
        self, content: str, url: str, doc_title: str
    ) -> List[Dict[str, Any]]:
        """
        Create recursive chunks from document content.

        Args:
            content: Document content to chunk
            url: Source URL
            doc_title: Document title

        Returns:
            List of chunk dictionaries
        """
        from utils.chunk_utils import recursive_chunk_text

        chunk_size = self.config.chunking.chunk_size
        chunk_overlap = self.config.chunking.chunk_overlap

        recursive_chunks = recursive_chunk_text(
            content, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        return [
            {
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
            for i, chunk_text in enumerate(recursive_chunks)
        ]

    def _save_chunks_to_file(
        self, chunks: List[Dict[str, Any]], output_file: str
    ) -> None:
        """
        Save chunks to a JSON file.

        Args:
            chunks: Chunks to save
            output_file: Output file name
        """
        if not chunks:
            return

        output_path = self.data_dir / output_file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")

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
            documents = self._load_documents_from_file(documents)

        # Ensure we have documents
        if not documents:
            logger.warning("No documents to chunk")
            return []

        logger.info(f"Chunking {len(documents)} documents")
        all_chunks = []

        # Get chunking strategy from config if not specified
        chunking_strategy = strategy or self.config.chunking.strategy

        # Process each document
        for doc in documents:
            url = doc.get("url", "")
            content = doc.get("content", "")
            doc_title = doc.get("title", "Untitled Document")

            if not content:
                continue

            try:
                if chunking_strategy == ChunkingStrategy.SEMANTIC:
                    chunks = self._create_semantic_chunks(content, url, doc_title)
                    all_chunks.extend(chunks)
                elif chunking_strategy == ChunkingStrategy.SLIDING_WINDOW:
                    chunks = self._create_sliding_window_chunks(content, url, doc_title)
                    all_chunks.extend(chunks)
                elif chunking_strategy == ChunkingStrategy.RECURSIVE:
                    chunks = self._create_recursive_chunks(content, url, doc_title)
                    all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error chunking document from {url}: {e}")
                continue

        # Save chunks if output file specified
        if output_file:
            self._save_chunks_to_file(all_chunks, output_file)

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
            chunk_list = chunks if isinstance(chunks, list) else []
            embedded_chunks = self.embedding_service.embed_chunks(chunk_list)

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
            # Ensure we return a list of dictionaries, not a string
            return [] if isinstance(chunks, str) else chunks

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
                    chunk_list = chunks if isinstance(chunks, list) else []
                    chunks = self.embedding_service.embed_chunks(chunk_list)

            except Exception as e:
                logger.error(f"Error loading embedded chunks from {chunks}: {e}")
                return False

        # Ensure we have chunks with embeddings
        chunk_list = chunks if isinstance(chunks, list) else []
        valid_chunks = [chunk for chunk in chunk_list if embedding_field in chunk]

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
        model: Optional[str] = "",
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
        # Ensure we're working with dictionaries by explicitly requesting as_dict=True
        # Convert SearchResult objects to dictionaries if needed
        context_str = "\n\n".join(
            f"DOCUMENT {i+1}:\n{result['content'] if isinstance(result, dict) else result.content}"
            for i, result in enumerate(results)
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

    def _execute_pipeline_step(
        self,
        step_name: str,
        step_fn: Callable,
        input_data: Any,
        depends_on_previous: bool,
        previous_successful: bool,
        result: Dict[str, Any],
    ) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Execute a pipeline step with standardized error handling and logging.

        Args:
            step_name: Name of the pipeline step
            step_fn: Function to execute for this step
            input_data: Input data for the step
            depends_on_previous: Whether this step depends on the previous step's success
            previous_successful: Whether the previous step was successful
            result: Current result dictionary to update

        Returns:
            Tuple of (success_flag, output_data, updated_result_dict)
        """
        # Skip step if it depends on a previous step that failed
        if depends_on_previous and not previous_successful:
            logger.error(f"No input data for {step_name} step")
            return self._extracted_from__execute_pipeline_step_27(result, step_name)
        try:
            # Execute the step function
            output_data = step_fn(input_data)

            # Check for empty results
            if not output_data:
                logger.error(
                    f"{step_name.capitalize()} step failed - no output created"
                )
                return self._handle_pipeline_step_failure(result, step_name)
            # Log success and update result
            if isinstance(output_data, list):
                logger.info(
                    f"{step_name.capitalize()} step completed: {len(output_data)} items"
                )
                result["document_counts"][step_name] = len(output_data)
            else:
                logger.info(f"{step_name.capitalize()} step completed successfully")

            result["steps"][step_name] = True
            return True, output_data, result

        except Exception as e:
            # Handle errors consistently
            logger.error(f"{step_name.capitalize()} step failed: {e}")
            return self._handle_pipeline_step_failure(result, step_name)

    def _handle_pipeline_step_failure(self, result, step_name):
        result["steps"][step_name] = False
        result["success"] = False
        return False, None, result

    def _get_default_input(self, step_name: str, output_file: str) -> Any:
        """
        Get default input for a pipeline step from file when the previous step was skipped.

        Args:
            step_name: Name of the step
            output_file: File to load data from

        Returns:
            File path as string to pass to the next step
        """
        return str(self.data_dir / output_file)

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
        batch_size: int = 50,
        max_memory_percent: float = 80.0,
        enable_backpressure: bool = True,
        enable_benchmarking: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline with backpressure mechanisms and benchmarking.

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
            batch_size: Maximum number of items to process in a batch
            max_memory_percent: Maximum memory usage percentage before throttling
            enable_backpressure: Whether to enable backpressure mechanisms
            enable_benchmarking: Whether to enable performance benchmarking

        Returns:
            Dictionary with pipeline status and document counts
        """
        import time
        from functools import wraps

        import psutil

        def benchmark(name):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    if not enable_benchmarking:
                        return func(*args, **kwargs)

                    start_time = time.time()
                    start_memory = psutil.Process().memory_info().rss / (
                        1024 * 1024
                    )  # MB

                    result = func(*args, **kwargs)

                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / (
                        1024 * 1024
                    )  # MB

                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory

                    logger.info(
                        f"BENCHMARK - {name}: Time={duration:.2f}s, Memory Delta={memory_delta:.2f}MB"
                    )

                    if "benchmarks" not in pipeline_result:
                        pipeline_result["benchmarks"] = {}

                    pipeline_result["benchmarks"][name] = {
                        "duration_seconds": round(duration, 2),
                        "memory_delta_mb": round(memory_delta, 2),
                        "items_processed": (
                            getattr(result, "__len__", lambda: 1)() if result else 0
                        ),
                    }

                    return result

                return wrapper

            return decorator

        def apply_backpressure(current_batch_size):
            if not enable_backpressure:
                return current_batch_size

            memory_percent = psutil.virtual_memory().percent

            if memory_percent > max_memory_percent:
                new_batch_size = max(1, int(current_batch_size * 0.5))
                logger.warning(
                    f"High memory usage ({memory_percent:.1f}%), reducing batch size from {current_batch_size} to {new_batch_size}"
                )
                import gc

                gc.collect()
                time.sleep(1)
                return new_batch_size
            if memory_percent < max_memory_percent * 0.7:
                new_batch_size = min(batch_size * 2, 500)  # Cap at 500
                if new_batch_size > current_batch_size:
                    logger.info(
                        f"Low memory usage ({memory_percent:.1f}%), increasing batch size from {current_batch_size} to {new_batch_size}"
                    )
                    return new_batch_size

            return current_batch_size

        # Batch processing function
        def process_in_batches(items, process_func, batch_size_initial=batch_size):
            if not items:
                return []

            results = []
            current_batch_size = batch_size_initial

            for i in range(0, len(items), current_batch_size):
                batch = items[i : i + current_batch_size]
                logger.info(
                    f"Processing batch {i//current_batch_size + 1}/{(len(items) + current_batch_size - 1)//current_batch_size} ({len(batch)} items)"
                )

                # Process the batch
                batch_results = process_func(batch)

                if batch_results:
                    results.extend(batch_results)

                current_batch_size = apply_backpressure(current_batch_size)

                if i + current_batch_size < len(items):
                    time.sleep(0.5)

            return results

        logger.info("Starting RAG pipeline with performance optimization")
        pipeline_result = {
            "success": True,
            "steps": {},
            "timestamp": datetime.now().isoformat(),
            "document_counts": {},
        }

        # Initialize data variables
        documents = None
        chunks = None
        embedded_chunks = None
        extract_success = False
        chunk_success = False
        embed_success = False

        # Step 1: Extract content
        if run_extract:
            extract_params = {
                "url": url,
                "urls": urls,
                "sitemap_url": sitemap_url,
                "links_file": links_file,
                "limit": limit,
                "output_file": "raw_documents.json",
                "output_format": output_format,
            }

            # Define the extract function to pass to _execute_pipeline_step
            @benchmark("extract_content")
            def extract_fn(_):
                return self.extract_content(**extract_params)

            extract_success, documents, pipeline_result = self._execute_pipeline_step(
                "documents", extract_fn, None, False, True, pipeline_result
            )

            if not extract_success and run_chunk:
                return pipeline_result

        # Step 2: Chunk documents
        if run_chunk:
            # If we didn't run extract, try to load documents from file
            if not documents and not run_extract:
                documents = self._get_default_input("documents", "raw_documents.json")

            # Define the chunking function with batching
            @benchmark("chunk_documents")
            def chunk_fn(docs):
                if (
                    enable_backpressure
                    and isinstance(docs, list)
                    and len(docs) > batch_size
                ):
                    # Process documents in batches
                    def process_batch(batch):
                        return self.chunk_documents(documents=batch, output_file=None)

                    all_chunks = process_in_batches(docs, process_batch)

                    if all_chunks:
                        self._save_chunks_to_file(all_chunks, "document_chunks.json")

                    return all_chunks
                # Process all documents at once if backpressure is disabled or batch is small
                return self.chunk_documents(
                    documents=docs, output_file="document_chunks.json"
                )

            chunk_success, chunks, pipeline_result = self._execute_pipeline_step(
                "chunks",
                chunk_fn,
                documents,
                run_extract,
                extract_success,
                pipeline_result,
            )

            if not chunk_success and run_embed:
                return pipeline_result

        # Step 3: Embed chunks
        if run_embed:
            # If we didn't run chunk, try to load chunks from file
            if not chunks and not run_chunk:
                chunks = self._get_default_input("chunks", "document_chunks.json")

            # Define the embedding function with batching
            @benchmark("embed_chunks")
            def embed_fn(chunk_data):
                if (
                    enable_backpressure
                    and isinstance(chunk_data, list)
                    and len(chunk_data) > batch_size
                ):
                    # Process chunks in batches
                    def process_batch(batch):
                        return self.embed_chunks(chunks=batch, output_file=None)

                    all_embedded = process_in_batches(chunk_data, process_batch)

                    if all_embedded:
                        self._save_chunks_to_file(all_embedded, "embedded_chunks.json")

                    return all_embedded
                # Process all chunks at once if backpressure is disabled or batch is small
                return self.embed_chunks(
                    chunks=chunk_data, output_file="embedded_chunks.json"
                )

            (
                embed_success,
                embedded_chunks,
                pipeline_result,
            ) = self._execute_pipeline_step(
                "embedded_chunks",
                embed_fn,
                chunks,
                run_chunk,
                chunk_success,
                pipeline_result,
            )

            if not embed_success and run_store:
                return pipeline_result

        # Step 4: Store chunks
        if run_store:
            # If we didn't run embed, try to load embedded chunks from file
            if not embedded_chunks and not run_embed:
                embedded_chunks = self._get_default_input(
                    "embedded_chunks", "embedded_chunks.json"
                )

            # Define the storage function with batching
            @benchmark("store_chunks")
            def store_fn(embeds):
                if (
                    enable_backpressure
                    and isinstance(embeds, list)
                    and len(embeds) > batch_size
                ):
                    # Process embedded chunks in batches
                    def process_batch(batch):
                        return self.store_chunks(batch)

                    # Store in batches but don't collect results (just success/failure)
                    success = True
                    for i in range(0, len(embeds), batch_size):
                        batch = embeds[i : i + batch_size]
                        logger.info(
                            f"Storing batch {i//batch_size + 1}/{(len(embeds) + batch_size - 1)//batch_size} ({len(batch)} items)"
                        )
                        batch_success = process_batch(batch)
                        if not batch_success:
                            success = False

                        apply_backpressure(batch_size)

                    if success:
                        # Count documents in vector store
                        doc_count = self.vector_store.count_documents()
                        pipeline_result["document_counts"]["stored_vectors"] = doc_count

                    return success
                # Process all embedded chunks at once if backpressure is disabled or batch is small
                success = self.store_chunks(embeds)
                if success:
                    # Count documents in vector store
                    doc_count = self.vector_store.count_documents()
                    pipeline_result["document_counts"]["stored_vectors"] = doc_count
                return success

            store_success, _, pipeline_result = self._execute_pipeline_step(
                "store",
                store_fn,
                embedded_chunks,
                run_embed,
                embed_success,
                pipeline_result,
            )

        if enable_benchmarking and "benchmarks" in pipeline_result:
            total_duration = sum(
                b["duration_seconds"] for b in pipeline_result["benchmarks"].values()
            )
            pipeline_result["benchmarks"]["total"] = {
                "duration_seconds": round(total_duration, 2),
                "steps_executed": len(pipeline_result["benchmarks"]),
            }

            logger.info(f"RAG pipeline completed in {total_duration:.2f} seconds")
        else:
            logger.info("RAG pipeline completed successfully")

        return pipeline_result

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
        if path := parsed_url.path.strip("/"):
            return path.split("/")[-1].replace("-", " ").replace("_", " ").title()
        return parsed_url.netloc


# Singleton instance for easy access
_default_pipeline = None


def get_pipeline(
    collection_name: Optional[str] = None,
    embedding_model_type: Optional[Union[str, EmbeddingModelType]] = None,
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
