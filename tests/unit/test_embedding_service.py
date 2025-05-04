import sys
import numpy as np
import pytest
from pathlib import Path

# Add the module paths directly
project_root = Path(__file__).parent.parent.parent
config_path = project_root / "RAGnificent" / "core"
embedding_path = project_root / "RAGnificent" / "rag"

# Insert paths for direct imports
sys.path.insert(0, str(config_path.parent))
sys.path.insert(0, str(embedding_path.parent))

# Direct imports from the module files
from core.config import EmbeddingConfig
from rag.embedding import EmbeddingService


class TestEmbeddingService:
    @pytest.fixture
    def embedding_service(self):
        return EmbeddingService(EmbeddingConfig(model_name="all-MiniLM-L6-v2"))

    def test_embed_texts(self, embedding_service):
        texts = ["test sentence", "another test"]
        embeddings = embedding_service.embed_texts(texts)

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 2
        assert embeddings[0].shape == (384,)

    def test_store_and_search(self, embedding_service):
        texts = ["apple fruit", "banana fruit", "car vehicle"]
        metadata = [{"type": "fruit"}, {"type": "fruit"}, {"type": "vehicle"}]

        ids = embedding_service.store_embeddings(texts, metadata)
        results = embedding_service.search_similar("fruit", k=2)

        assert len(ids) == 3
        assert len(results) == 2
        assert all("fruit" in r["payload"]["type"] for r in results)
