"""
RAG Pipeline module for RAGnificent.

Provides a complete end-to-end pipeline for Retrieval Augmented Generation,
combining Rust-optimized scraping with embedding, vector storage, and search.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Use relative imports for internal modules
# Import fix applied
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ChunkingStrategy, get_config
from core.scraper import MarkdownScraper
from rag.embedding import get_embedding_service
from rag.search import SearchResult, get_search
from rag.vector_store import get_vector_store
from utils.chunk_utils import ContentChunker

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
            html_content = self.scraper.scrape_website(url, skip_cache=skip_cache)
            content = self.scraper.convert_html(html_content, url, output_format)

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
            return parser.parse_sitemap(sitemap_url, limit=limit)
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
        documents = []

        # Process single URL
        if url:
            logger.info(f"Extracting content from URL: {url}")
            if document := self._process_single_url(url, output_format, skip_cache):
                documents.append(document)

        elif urls:
            logger.info(f"Extracting content from {len(urls)} URLs")
            documents = self._process_url_list(urls, limit, output_format, skip_cache)

        elif sitemap_url:
            logger.info(f"Extracting content from sitemap: {sitemap_url}")
            if sitemap_urls := self._get_urls_from_sitemap(sitemap_url, limit):
                documents = self._process_url_list(
                    sitemap_urls, None, output_format, skip_cache
                )

        elif links_file:
            logger.info(f"Extracting content from links in file: {links_file}")
            if file_urls := self._get_urls_from_file(links_file, limit):
                documents = self._process_url_list(
                    file_urls, None, output_format, skip_cache
                )

        # Save raw documents if output file specified
        if output_file:
            self._save_documents(documents, output_file)

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
                return self._extracted_from__execute_pipeline_step_27(result, step_name)
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
            return self._extracted_from__execute_pipeline_step_27(result, step_name)

    # TODO Rename this here and in `_execute_pipeline_step`
    def _extracted_from__execute_pipeline_step_27(self, result, step_name):
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
            def extract_fn(_):
                return self.extract_content(**extract_params)

            extract_success, documents, result = self._execute_pipeline_step(
                "documents", extract_fn, None, False, True, result
            )

            if not extract_success and run_chunk:
                return result

        # Step 2: Chunk documents
        if run_chunk:
            # If we didn't run extract, try to load documents from file
            if not documents and not run_extract:
                documents = self._get_default_input("documents", "raw_documents.json")

            # Define the chunking function
            def chunk_fn(docs):
                return self.chunk_documents(
                    documents=docs, output_file="document_chunks.json"
                )

            chunk_success, chunks, result = self._execute_pipeline_step(
                "chunks", chunk_fn, documents, run_extract, extract_success, result
            )

            if not chunk_success and run_embed:
                return result

        # Step 3: Embed chunks
        if run_embed:
            # If we didn't run chunk, try to load chunks from file
            if not chunks and not run_chunk:
                chunks = self._get_default_input("chunks", "document_chunks.json")

            # Define the embedding function
            def embed_fn(chunk_data):
                return self.embed_chunks(
                    chunks=chunk_data, output_file="embedded_chunks.json"
                )

            embed_success, embedded_chunks, result = self._execute_pipeline_step(
                "embedded_chunks", embed_fn, chunks, run_chunk, chunk_success, result
            )

            if not embed_success and run_store:
                return result

        # Step 4: Store chunks
        if run_store:
            # If we didn't run embed, try to load embedded chunks from file
            if not embedded_chunks and not run_embed:
                embedded_chunks = self._get_default_input(
                    "embedded_chunks", "embedded_chunks.json"
                )

            # Define the storage function
            def store_fn(embeds):
                success = self.store_chunks(embeds)
                if success:
                    # Count documents in vector store
                    doc_count = self.vector_store.count_documents()
                    result["document_counts"]["stored_vectors"] = doc_count
                return success

            store_success, _, result = self._execute_pipeline_step(
                "store", store_fn, embedded_chunks, run_embed, embed_success, result
            )

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
        if path := parsed_url.path.strip("/"):
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
