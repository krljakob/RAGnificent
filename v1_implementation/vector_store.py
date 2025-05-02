import logging
from typing import Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import QdrantConfig


class QdrantVectorStore:
    def __init__(self, config: Optional[QdrantConfig] = None):
        """Initialize with optional config (defaults will be loaded)"""
        self.config = config or QdrantConfig()
        if self.config.host == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                https=self.config.https,
                api_key=self.config.api_key,
                prefer_grpc=True,
                timeout=self.config.timeout
            )
        self.collection_name = self.config.collection
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.config.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
        except Exception as e:
            logging.error(f"Failed to ensure collection: {e}")
            raise

    def store_embeddings(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict],
        batch_size: int = 100
    ):
        """Store embeddings in batches"""
        import uuid
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(idx))),
                vector=embedding.tolist(),
                payload=metadata
            )
            for idx, embedding, metadata in zip(ids, embeddings, metadatas, strict=False)
        ]

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            except Exception as e:
                logging.error(f"Batch {i//batch_size} failed: {e}")
                raise

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar vectors"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=models.Filter(**filter) if filter else None,
                limit=k
            )
            return [{
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            } for hit in results]
        except Exception as e:
            logging.error(f"Search failed: {e}")
            raise

    def delete_collection(self):
        """Clean up collection"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logging.error(f"Deletion failed: {e}")
            raise
