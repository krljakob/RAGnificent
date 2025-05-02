import logging
from functools import lru_cache
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig
from .vector_store import QdrantVectorStore


class EmbeddingService:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.model = self._load_model()
        self.vector_store = QdrantVectorStore()

    @lru_cache(maxsize=1)
    def _load_model(self):
        """Cache the model loading"""
        try:
            return SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Batch embed texts with error handling"""
        if not texts:
            return np.array([])

        try:
            return self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        except Exception as e:
            logging.error(f"Embedding failed: {e}")
            raise

    def store_embeddings(self, texts: List[str], metadata: List[dict]) -> List[str]:
        """Full pipeline: embed texts and store in vector DB"""
        embeddings = self.embed_texts(texts)
        ids = [str(hash(text)) for text in texts]  # Simple deterministic ID generation

        self.vector_store.store_embeddings(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadata
        )
        return ids

    def search_similar(self, query: str, k: int = 5) -> List[dict]:
        """Search for similar texts"""
        query_embedding = self.embed_texts([query])[0]
        return self.vector_store.search(query_embedding, k=k)
