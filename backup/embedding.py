"""
Embedding module for RAGnificent.
Provides functions for embedding text chunks using various models.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

# Use v1_implementation's embedding functionality
from v1_implementation.embedding import (
    get_embedding_model,
    get_embedding,
    embed_text,
    embed_texts_batched,
)
from v1_implementation.config import get_config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings for text chunks."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service.

        Args:
            model_name: The name of the embedding model to use.
                        If None, uses the model from config.
        """
        self.config = get_config()
        self.model_name = model_name or self.config.get("embedding", {}).get(
            "model", "BAAI/bge-small-en-v1.5"
        )
        
        try:
            self.model = get_embedding_model(self.model_name)
            logger.info(f"Initialized embedding service with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise
    
    def embed_chunk(self, chunk: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an embedding for a single chunk.

        Args:
            chunk: Either a string text chunk or a Chunk object dictionary

        Returns:
            The chunk with an added embedding field
        """
        if isinstance(chunk, str):
            text = chunk
            result = {"content": text, "embedding": embed_text(text, self.model)}
            return result
            
        # If it's a chunk object
        try:
            content = chunk.get("content", "") 
            if not content and "text" in chunk:
                content = chunk["text"]
                
            # Generate embedding
            embedding = embed_text(content, self.model)
            
            # Add embedding to the chunk
            result = {**chunk, "embedding": embedding}
            return result
            
        except Exception as e:
            logger.error(f"Error embedding chunk: {e}")
            # Return original chunk without embedding in case of error
            return chunk
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple chunks with batching for efficiency.

        Args:
            chunks: List of chunk objects

        Returns:
            List of chunks with embeddings added
        """
        try:
            # Extract text content for batched embedding
            texts = []
            for chunk in chunks:
                content = chunk.get("content", "")
                if not content and "text" in chunk:
                    content = chunk["text"]
                texts.append(content)
                
            # Generate embeddings in batch
            embeddings = embed_texts_batched(texts, self.model)
            
            # Add embeddings back to chunks
            result = []
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    result.append({**chunk, "embedding": embeddings[i]})
                else:
                    # This should not happen if batching works correctly
                    logger.warning(f"Missing embedding for chunk {i}")
                    result.append(chunk)
                    
            return result
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            return chunks  # Return original chunks without embeddings in case of error

# Singleton instance for easy access
default_embedding_service = None

def get_embedding_service(model_name: Optional[str] = None) -> EmbeddingService:
    """
    Get or create the default embedding service.

    Args:
        model_name: The name of the embedding model to use

    Returns:
        The embedding service instance
    """
    global default_embedding_service
    if default_embedding_service is None:
        default_embedding_service = EmbeddingService(model_name)
    return default_embedding_service
