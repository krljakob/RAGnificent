import types
from datetime import datetime

import pytest


def test_get_urls_from_sitemap_uses_package_imports(monkeypatch):
    from RAGnificent.rag.pipeline import Pipeline

    # fake SitemapURL-like objects with .loc attribute
    class URLObj:
        def __init__(self, loc):
            self.loc = loc

    class FakeSitemapParser:
        def __init__(self, *args, **kwargs):
            pass

        def parse_sitemap(self, url):
            return [URLObj("https://example.com/a"), URLObj("https://example.com/b")]

    # patch the class in the utils module so the local import resolves to this
    import RAGnificent.utils.sitemap_utils as sm

    monkeypatch.setattr(sm, "SitemapParser", FakeSitemapParser, raising=True)

    p = Pipeline()
    urls = p._get_urls_from_sitemap("https://example.com/sitemap.xml", limit=1)
    assert urls == ["https://example.com/a"]


def test_chunk_helpers_route_to_utils(monkeypatch):
    # patch chunking helpers to ensure routing works
    import RAGnificent.utils.chunk_utils as cu
    from RAGnificent.rag.pipeline import Pipeline

    monkeypatch.setattr(
        cu,
        "chunk_text",
        lambda content, chunk_size, chunk_overlap: ["X", "Y"],
        raising=True,
    )
    monkeypatch.setattr(
        cu,
        "recursive_chunk_text",
        lambda content, chunk_size, chunk_overlap: ["R1", "R2", "R3"],
        raising=True,
    )

    p = Pipeline()
    sw = p._create_sliding_window_chunks("content", url="https://x", doc_title="T")
    rc = p._create_recursive_chunks("content", url="https://x", doc_title="T")

    assert len(sw) == 2 and all(c["chunk_type"] == "sliding_window" for c in sw)
    assert len(rc) == 3 and all(c["chunk_type"] == "recursive" for c in rc)
