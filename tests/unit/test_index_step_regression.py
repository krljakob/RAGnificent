import os
from pathlib import Path

import numpy as np
import pytest


def test_execute_index_step_embeds_and_stores(tmp_path, monkeypatch):
    # Arrange: create a simple markdown file in a temp data dir
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    md_file = data_dir / "doc.md"
    md_file.write_text("# Title\n\nHello world paragraph.", encoding="utf-8")

    # Dummy vector store that just records documents
    class DummyVS:
        def __init__(self, *args, **kwargs):
            self.stored = []

        def store_documents(self, docs, **kwargs):
            # Ensure embeddings present
            for d in docs:
                assert "embedding" in d
                emb = d["embedding"]
                # Accept numpy array or list
                assert isinstance(emb, (list, np.ndarray))
            self.stored.extend(docs)
            return True

        def count_documents(self):
            return len(self.stored)

    # Dummy search object used by pipeline
    class DummySearch:
        def __init__(self, *args, **kwargs):
            pass

    import RAGnificent.rag.pipeline as pl

    # Patch to avoid real Qdrant and search init
    monkeypatch.setattr(pl, "get_vector_store", lambda *a, **k: DummyVS())
    monkeypatch.setattr(pl, "get_search", lambda *a, **k: DummySearch())

    # Act: construct pipeline with simple embedder to avoid heavy models
    pipeline = pl.Pipeline(
        data_dir=str(data_dir),
        embedding_model_type="simpler",
        continue_on_error=True,
    )

    result = pipeline._execute_index_step({"input_dir": "."})

    # Assert
    assert isinstance(result, dict)
    assert result.get("indexed_documents", 0) >= 1
