"""
FastAPI application for RAGnificent.

Provides RESTful API endpoints for scraping, conversion, indexing, and search.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from RAGnificent.core.async_scraper import AsyncMarkdownScraper
from RAGnificent.core.logging import get_logger
from RAGnificent.rag.pipeline import Pipeline
from RAGnificent.rag.search import SearchResult

logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title="RAGnificent API",
    description="A powerful RAG pipeline API for web scraping, content processing, and semantic search",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ScrapeRequest(BaseModel):
    """Request model for scraping operations."""

    url: str = Field(..., description="URL to scrape")
    format: str = Field("markdown", description="Output format (markdown, json, xml)")
    use_sitemap: bool = Field(False, description="Use sitemap for URL discovery")
    save_chunks: bool = Field(False, description="Save content chunks for RAG")
    chunk_size: int = Field(1000, description="Maximum chunk size in characters")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    parallel: bool = Field(True, description="Enable parallel processing")
    max_workers: int = Field(4, description="Maximum number of parallel workers")


class ScrapeResponse(BaseModel):
    """Response model for scraping operations."""

    success: bool
    message: str
    scraped_urls: List[str]
    total_documents: int
    output_dir: Optional[str] = None


class ConvertRequest(BaseModel):
    """Request model for content conversion."""

    content: str = Field(..., description="Content to convert")
    from_format: str = Field("html", description="Input format")
    to_format: str = Field("markdown", description="Output format")
    chunk: bool = Field(False, description="Enable chunking")
    chunk_size: int = Field(1000, description="Chunk size")
    chunk_overlap: int = Field(200, description="Chunk overlap")


class ConvertResponse(BaseModel):
    """Response model for content conversion."""

    success: bool
    message: str
    converted_content: str
    chunks: Optional[List[Dict]] = None


class SearchRequest(BaseModel):
    """Request model for search operations."""

    query: str = Field(..., description="Search query")
    collection_name: Optional[str] = Field(None, description="Collection to search in")
    top_k: int = Field(5, description="Number of results to return")
    threshold: float = Field(0.7, description="Minimum similarity threshold")


class SearchResponse(BaseModel):
    """Response model for search operations."""

    success: bool
    message: str
    query: str
    results: List[Dict]
    total_results: int


class PipelineRequest(BaseModel):
    """Request model for pipeline execution."""

    config: Dict = Field(..., description="Pipeline configuration")
    continue_on_error: bool = Field(False, description="Continue on errors")


class PipelineResponse(BaseModel):
    """Response model for pipeline execution."""

    success: bool
    message: str
    pipeline_name: str
    steps_executed: int
    results: List[Dict]


# Global pipeline instance (consider using dependency injection in production)
_pipeline = None


def get_pipeline() -> Pipeline:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = Pipeline()
            logger.info("Initialized pipeline instance")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to initialize pipeline"
            ) from e
    return _pipeline


# API Endpoints


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to RAGnificent API",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "operational",
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    try:
        # Basic health checks
        pipeline = get_pipeline()
        return {
            "status": "healthy",
            "pipeline": "initialized",
            "timestamp": str(asyncio.get_event_loop().time()),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy") from e


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_endpoint(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Scrape a website and convert content to the specified format.

    Supports both single URL and sitemap-based scraping.
    """
    try:
        logger.info(f"Scraping request for URL: {request.url}")

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)

            async with AsyncMarkdownScraper(
                max_workers=request.max_workers,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            ) as scraper:

                if request.use_sitemap:
                    scraped_urls = await scraper.scrape_by_sitemap(
                        request.url,
                        output_dir=str(output_dir),
                        output_format=request.format,
                        save_chunks=request.save_chunks,
                    )
                else:
                    await scraper._scrape_and_save(
                        request.url,
                        str(output_dir),
                        request.format,
                        request.save_chunks,
                    )
                    scraped_urls = [request.url]

            # Count output files
            output_files = list(output_dir.glob(f"*.{request.format}"))

            return ScrapeResponse(
                success=True,
                message=f"Successfully scraped {len(scraped_urls)} URLs",
                scraped_urls=scraped_urls,
                total_documents=len(output_files),
                output_dir=str(output_dir) if request.save_chunks else None,
            )

    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}") from e


@app.post("/convert", response_model=ConvertResponse)
async def convert_endpoint(request: ConvertRequest):
    """
    Convert content between different formats.

    Supports conversion from HTML to markdown, JSON, XML and chunking.
    """
    try:
        logger.info(
            f"Converting content from {request.from_format} to {request.to_format}"
        )

        if request.from_format == "html":
            # Use Rust converter if available
            try:
                from RAGnificent.ragnificent_rs import (
                    convert_to_json,
                    convert_to_markdown,
                    convert_to_xml,
                )

                if request.to_format == "markdown":
                    converted = convert_to_markdown(request.content)
                elif request.to_format == "json":
                    converted = convert_to_json(request.content)
                elif request.to_format == "xml":
                    converted = convert_to_xml(request.content)
                else:
                    raise ValueError(f"Unsupported output format: {request.to_format}")

            except ImportError as e:
                # Fallback to Python implementation
                from markdownify import markdownify

                if request.to_format == "markdown":
                    converted = markdownify(request.content, heading_style="ATX")
                else:
                    raise HTTPException(
                        status_code=501,
                        detail="Format conversion not available without Rust components",
                    ) from e
        else:
            # For non-HTML inputs, return as-is for now
            converted = request.content

        # Apply chunking if requested
        chunks = None
        if request.chunk:
            from RAGnificent.utils.chunk_utils import ContentChunker

            chunker = ContentChunker(request.chunk_size, request.chunk_overlap)
            chunk_results = chunker.create_chunks_from_markdown(
                converted, source_url="api_request"
            )
            chunks = [
                {"content": chunk["content"], "metadata": chunk["metadata"]}
                for chunk in chunk_results
            ]

        return ConvertResponse(
            success=True,
            message=f"Successfully converted content from {request.from_format} to {request.to_format}",
            converted_content=converted,
            chunks=chunks,
        )

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Conversion failed: {str(e)}"
        ) from e


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Search the vector database for relevant content.

    Performs semantic search using embeddings.
    """
    try:
        logger.info(f"Search request for query: {request.query}")

        pipeline = get_pipeline()

        # Override collection if specified
        if request.collection_name:
            # Create temporary search instance for different collection
            from RAGnificent.rag.search import get_search

            search_service = get_search(request.collection_name)
        else:
            search_service = pipeline.search

        # Perform search
        results = search_service.search(
            request.query, top_k=request.top_k, threshold=request.threshold
        )

        formatted_results = [
            {
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata,
                "source": result.metadata.get("source_url", "unknown"),
            }
            for result in results
        ]
        return SearchResponse(
            success=True,
            message=f"Found {len(results)} results",
            query=request.query,
            results=formatted_results,
            total_results=len(results),
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@app.post("/query", response_model=Dict)
async def query_endpoint(
    query: str = Query(..., description="Search query"),
    collection: Optional[str] = Query(None, description="Collection name"),
    top_k: int = Query(5, description="Number of results"),
    use_llm: bool = Query(False, description="Use LLM for response generation"),
):
    """
    Advanced query endpoint with optional LLM integration.

    Can perform simple search or RAG-enhanced response generation.
    """
    try:
        logger.info(f"Query request: {query}")

        # Perform search first
        search_request = SearchRequest(
            query=query, collection_name=collection, top_k=top_k, threshold=0.6
        )
        search_response = await search_endpoint(search_request)

        if not use_llm:
            return search_response.dict()

        # Generate LLM response using search results
        try:
            import openai

            # Combine search results into context
            context = "\n\n".join(
                [
                    f"Source: {result['source']}\nContent: {result['content']}"
                    for result in search_response.results[:3]  # Use top 3 results
                ]
            )

            # Generate response (requires OpenAI API key)
            # This is a placeholder - implement according to your LLM setup
            llm_response = f"Based on the search results, here's what I found about '{query}':\n\n{context}"

            return {
                "success": True,
                "query": query,
                "llm_response": llm_response,
                "search_results": search_response.results,
                "sources": [r["source"] for r in search_response.results],
            }

        except ImportError:
            logger.warning("OpenAI not available, returning search results only")
            return search_response.dict()

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}") from e


@app.post("/pipeline", response_model=PipelineResponse)
async def pipeline_endpoint(
    request: PipelineRequest, background_tasks: BackgroundTasks
):
    """
    Execute a pipeline configuration.

    Runs a multi-step RAG pipeline as defined in the configuration.
    """
    try:
        logger.info("Pipeline execution request")

        # Create pipeline from config
        pipeline = Pipeline(
            config=request.config, continue_on_error=request.continue_on_error
        )

        # Execute pipeline steps
        results = []
        steps_executed = 0

        for step_result in pipeline.execute():
            results.append(step_result)
            if step_result["status"] == "success":
                steps_executed += 1

            # Break on error if not continuing
            if step_result["status"] == "error" and not request.continue_on_error:
                break

        pipeline_name = request.config.get("name", "Unnamed Pipeline")

        return PipelineResponse(
            success=True,
            message=f"Pipeline '{pipeline_name}' executed {steps_executed} steps",
            pipeline_name=pipeline_name,
            steps_executed=steps_executed,
            results=results,
        )

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Pipeline execution failed: {str(e)}"
        ) from e


@app.get("/collections", response_model=List[str])
async def list_collections():
    """List available vector collections."""
    try:
        # This would need to be implemented based on your vector store
        # For now, return a placeholder
        return ["default", "documents", "web_content"]
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collections") from e


@app.get("/stats", response_model=Dict)
async def get_stats():
    """Get system statistics and health metrics."""
    try:
        pipeline = get_pipeline()

        # Collect various stats
        return {
            "pipeline_initialized": _pipeline is not None,
            "scraper_stats": {},  # Could add scraper statistics
            "vector_store_stats": {},  # Could add vector store statistics
            "system_info": {"python_version": "3.12+", "api_version": "2.0.0"},
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics") from e


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "detail": "The requested endpoint does not exist",
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
