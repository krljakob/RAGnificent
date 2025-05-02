import logging
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantConfig(BaseSettings):
    host: str = Field(":memory:", env="QDRANT_HOST")
    port: int = Field(0, env="QDRANT_PORT")  # Not used for in-memory
    collection: str = Field("rag_vectors", env="QDRANT_COLLECTION")
    api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    https: bool = Field(False, env="QDRANT_HTTPS")
    vector_size: int = Field(384, env="QDRANT_VECTOR_SIZE")
    timeout: int = Field(10, env="QDRANT_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

class EmbeddingConfig(BaseSettings):
    model_name: str = Field("all-MiniLM-L6-v2")
    batch_size: int = Field(32)
    device: str = Field("cpu")

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8"
    )

class AppConfig:
    def __init__(self):
        self.qdrant = QdrantConfig()
        self.embedding = EmbeddingConfig()

        # Validate configs
        self._validate()

    def _validate(self):
        """Validate configuration values"""
        if self.qdrant.vector_size <= 0:
            raise ValueError("Vector size must be positive")

        if not 0 < self.embedding.batch_size <= 256:
            raise ValueError("Batch size must be between 1 and 256")

def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration with optional env file override"""
    try:
        if config_path:
            if not config_path.exists():
                logging.warning(f"Config file not found at {config_path}")
            else:
                QdrantConfig.Config.env_file = str(config_path)
                EmbeddingConfig.Config.env_file = str(config_path)

        return AppConfig()
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        raise
