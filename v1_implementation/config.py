#!/usr/bin/env python
"""
Configuration module for RAG implementation

Centralized configuration management for all RAG components with
environment variable integration, validation, and proper defaults.
"""
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging for this module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define global constants
DATA_DIR = Path(__file__).parent.parent / 'data'
MODELS_DIR = Path(__file__).parent.parent / 'models'

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class LogLevel(str, Enum):
    """Log levels enum for type safety"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EmbeddingModelType(str, Enum):
    """Supported embedding model types"""
    OPENAI = "openai"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    TFIDF = "tfidf"
    SIMPLER = "simpler"  # Fallback basic embedding


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies"""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"


class QdrantConfig(BaseSettings):
    """Qdrant vector database configuration"""
    host: str = Field(":memory:", description="Qdrant server host or :memory: for in-memory", env="QDRANT_HOST")
    port: int = Field(6333, description="Qdrant server port", env="QDRANT_PORT")
    collection: str = Field("rag_documents", description="Qdrant collection name", env="QDRANT_COLLECTION")
    api_key: Optional[str] = Field(None, description="Qdrant API key", env="QDRANT_API_KEY")
    https: bool = Field(False, description="Use HTTPS for Qdrant connection", env="QDRANT_HTTPS")
    vector_size: int = Field(384, description="Vector size for embeddings", env="QDRANT_VECTOR_SIZE")
    timeout: int = Field(10, description="Qdrant client timeout in seconds", env="QDRANT_TIMEOUT")
    prefer_grpc: bool = Field(True, description="Use gRPC protocol for better performance", env="QDRANT_PREFER_GRPC")

    @field_validator('vector_size')
    def check_vector_size(cls, v):
        """Validate vector size is positive"""
        if v <= 0:
            raise ValueError("Vector size must be positive")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration"""
    api_key: Optional[str] = Field(None, description="OpenAI API key", env="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model name", env="OPENAI_EMBEDDING_MODEL")
    completion_model: str = Field("gpt-3.5-turbo", description="OpenAI completion model name", env="OPENAI_COMPLETION_MODEL")
    max_tokens: int = Field(1000, description="Maximum tokens for completions", env="OPENAI_MAX_TOKENS")
    temperature: float = Field(0.7, description="Temperature for completions", env="OPENAI_TEMPERATURE")
    request_timeout: int = Field(30, description="Request timeout in seconds", env="OPENAI_REQUEST_TIMEOUT")
    max_retries: int = Field(3, description="Maximum retries for API requests", env="OPENAI_MAX_RETRIES")

    @field_validator('temperature')
    def check_temperature(cls, v):
        """Validate temperature is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class EmbeddingConfig(BaseSettings):
    """Embedding configuration"""
    model_type: EmbeddingModelType = Field(
        EmbeddingModelType.SENTENCE_TRANSFORMER,
        description="Embedding model type to use",
        env="EMBEDDING_MODEL_TYPE"
    )
    model_name: str = Field("all-MiniLM-L6-v2", description="Model name for SentenceTransformer", env="EMBEDDING_MODEL_NAME")
    batch_size: int = Field(32, description="Batch size for embedding generation", env="EMBEDDING_BATCH_SIZE")
    device: str = Field("cpu", description="Device to use (cpu/cuda)", env="EMBEDDING_DEVICE")
    cache_dir: Optional[Path] = Field(None, description="Cache directory for embeddings", env="EMBEDDING_CACHE_DIR")
    use_cache: bool = Field(True, description="Whether to use embedding cache", env="EMBEDDING_USE_CACHE")

    @field_validator('batch_size')
    def check_batch_size(cls, v):
        """Validate batch size is reasonable"""
        if not 1 <= v <= 512:
            raise ValueError("Batch size must be between 1 and 512")
        return v

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class ChunkingConfig(BaseSettings):
    """Document chunking configuration"""
    strategy: ChunkingStrategy = Field(
        ChunkingStrategy.SEMANTIC,
        description="Chunking strategy to use",
        env="CHUNKING_STRATEGY"
    )
    chunk_size: int = Field(1000, description="Target chunk size in characters", env="CHUNKING_SIZE")
    chunk_overlap: int = Field(200, description="Overlap between chunks in characters", env="CHUNKING_OVERLAP")
    separator: str = Field("\n", description="Default separator for boundary decisions", env="CHUNKING_SEPARATOR")
    keep_separator: bool = Field(True, description="Whether to keep separators in chunks", env="CHUNKING_KEEP_SEPARATOR")

    @field_validator('chunk_size')
    def check_chunk_size(cls, v):
        """Validate chunk size is positive"""
        if v <= 0:
            raise ValueError("Chunk size must be positive")
        return v

    @field_validator('chunk_overlap')
    def check_chunk_overlap(cls, v, values):
        """Validate chunk overlap is less than chunk size"""
        if 'chunk_size' in values.data and v >= values.data['chunk_size']:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v

    model_config = SettingsConfigDict(
        env_prefix="CHUNKING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class ExtractionConfig(BaseSettings):
    """Document extraction configuration"""
    use_sitemap: bool = Field(True, description="Whether to use sitemap for URL discovery", env="EXTRACTION_USE_SITEMAP")
    follow_links: bool = Field(True, description="Whether to follow links during extraction", env="EXTRACTION_FOLLOW_LINKS")
    max_depth: int = Field(2, description="Maximum depth for link following", env="EXTRACTION_MAX_DEPTH")
    timeout: int = Field(10, description="Request timeout in seconds", env="EXTRACTION_TIMEOUT")
    user_agent: str = Field(
        "Mozilla/5.0 RAGnificent/1.0",
        description="User agent for web requests",
        env="EXTRACTION_USER_AGENT"
    )
    respect_robots_txt: bool = Field(True, description="Whether to respect robots.txt", env="EXTRACTION_RESPECT_ROBOTS")
    rate_limit: float = Field(0.5, description="Seconds between requests (rate limiting)", env="EXTRACTION_RATE_LIMIT")

    model_config = SettingsConfigDict(
        env_prefix="EXTRACTION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class SearchConfig(BaseSettings):
    """Search configuration"""
    max_results: int = Field(5, description="Maximum search results to return", env="SEARCH_MAX_RESULTS")
    score_threshold: float = Field(0.6, description="Minimum similarity score threshold", env="SEARCH_SCORE_THRESHOLD")
    use_hybrid_search: bool = Field(False, description="Whether to use hybrid search", env="SEARCH_USE_HYBRID")
    rate_limit_per_minute: int = Field(60, description="Maximum searches per minute", env="SEARCH_RATE_LIMIT")
    enable_caching: bool = Field(True, description="Whether to cache search results", env="SEARCH_ENABLE_CACHE")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds", env="SEARCH_CACHE_TTL")

    @field_validator('score_threshold')
    def check_score_threshold(cls, v):
        """Validate score threshold is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Score threshold must be between 0 and 1")
        return v

    model_config = SettingsConfigDict(
        env_prefix="SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: LogLevel = Field(LogLevel.INFO, description="Logging level", env="LOGGING_LEVEL")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
        env="LOGGING_FORMAT"
    )
    file: Optional[Path] = Field(None, description="Log file path", env="LOGGING_FILE")
    console: bool = Field(True, description="Whether to log to console", env="LOGGING_CONSOLE")

    model_config = SettingsConfigDict(
        env_prefix="LOGGING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@dataclass
class AppConfig:
    """Main application configuration"""
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    openai: OpenAIConfig
    chunking: ChunkingConfig
    extraction: ExtractionConfig
    search: SearchConfig
    logging: LoggingConfig

    # Path constants
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR

    def configure_logging(self):
        """Configure logging based on settings"""
        handlers = []

        if self.logging.console:
            handlers.append(logging.StreamHandler())

        if self.logging.file:
            os.makedirs(os.path.dirname(self.logging.file), exist_ok=True)
            handlers.append(logging.FileHandler(self.logging.file))

        logging.basicConfig(
            level=getattr(logging, self.logging.level.value),
            format=self.logging.format,
            handlers=handlers
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "qdrant": self.qdrant.model_dump(),
            "embedding": self.embedding.model_dump(),
            "openai": {**self.openai.model_dump(), "api_key": "[REDACTED]" if self.openai.api_key else None},
            "chunking": self.chunking.model_dump(),
            "extraction": self.extraction.model_dump(),
            "search": self.search.model_dump(),
            "logging": self.logging.model_dump(),
        }


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load configuration with optional env file override

    Args:
        config_path: Optional path to .env config file

    Returns:
        Configured AppConfig instance

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If config file doesn't exist
    """
    try:
        # If config path provided, load it first
        if config_path:
            config_path = Path(config_path) if isinstance(config_path, str) else config_path
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
            load_dotenv(config_path)
            logger.info(f"Loaded configuration from {config_path}")

        # Load all config components
        qdrant_config = QdrantConfig()
        embedding_config = EmbeddingConfig()
        openai_config = OpenAIConfig()
        chunking_config = ChunkingConfig()
        extraction_config = ExtractionConfig()
        search_config = SearchConfig()
        logging_config = LoggingConfig()

        # Create app config
        config = AppConfig(
            qdrant=qdrant_config,
            embedding=embedding_config,
            openai=openai_config,
            chunking=chunking_config,
            extraction=extraction_config,
            search=search_config,
            logging=logging_config
        )

        # Configure logging
        config.configure_logging()
        
        logger.info("Configuration loaded successfully")
        return config

    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Configuration file error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise
