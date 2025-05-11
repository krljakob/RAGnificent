"""
Test edge cases in the pipeline module (part 2).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

project_root = Path(__file__).parent.parent.parent
rag_path = project_root / "RAGnificent" / "rag"
sys.path.insert(0, str(project_root))

from RAGnificent.rag.pipeline import Pipeline, get_pipeline


class TestPipelineEdgeCasesPart2(unittest.TestCase):
    """Test edge cases for the RAG pipeline (part 2)."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        self.pipeline = Pipeline(
            collection_name="test_collection", data_dir=self.data_dir
        )

        self.test_files_dir = self.data_dir / "test_files"
        os.makedirs(self.test_files_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_pipeline_step_dependencies(self):
        """Test pipeline step dependencies and error propagation."""
        with mock.patch(
            "RAGnificent.rag.pipeline.Pipeline.extract_content"
        ) as mock_extract:
            mock_extract.return_value = []

            result = self.pipeline.run_pipeline(
                url="https://example.com", run_extract=True, run_chunk=True
            )

            self.assertFalse(result["success"], "Pipeline should fail if extract fails")
            self.assertFalse(
                result["steps"].get("documents", True),
                "Documents step should be marked as failed",
            )
            self.assertFalse(
                "chunks" in result["steps"], "Chunks step should not be executed"
            )

    def test_missing_sitemap(self):
        """Test handling of missing sitemap."""
        urls = self.pipeline._get_urls_from_sitemap(
            "https://example.com/nonexistent-sitemap.xml", None
        )
        self.assertEqual(len(urls), 0, "Non-existent sitemap should return empty list")

    def test_malformed_sitemap(self):
        """Test handling of malformed sitemap."""
        sitemap_path = self.test_files_dir / "malformed-sitemap.xml"
        with open(sitemap_path, "w") as f:
            f.write("<urlset>malformed xml</urlset>")

        with mock.patch(
            "RAGnificent.utils.sitemap_utils.SitemapParser.parse_sitemap"
        ) as mock_parse:
            mock_parse.side_effect = Exception("XML parsing error")

            urls = self.pipeline._get_urls_from_sitemap(
                "https://example.com/sitemap.xml", None
            )
            self.assertEqual(len(urls), 0, "Malformed sitemap should return empty list")

    def test_missing_links_file(self):
        """Test handling of missing links file."""
        urls = self.pipeline._get_urls_from_file(
            str(self.test_files_dir / "nonexistent.txt"), None
        )
        self.assertEqual(
            len(urls), 0, "Non-existent links file should return empty list"
        )

    def test_malformed_links_file(self):
        """Test handling of malformed links file."""
        links_path = self.test_files_dir / "malformed-links.txt"
        with open(links_path, "wb") as f:
            f.write(b"\x80\x81\x82")

        urls = self.pipeline._get_urls_from_file(str(links_path), None)
        self.assertEqual(len(urls), 0, "Malformed links file should return empty list")

    def test_query_with_context_no_openai(self):
        """Test query_with_context when OpenAI is not available."""
        with mock.patch.dict("sys.modules", {"openai": None}):
            result = self.pipeline.query_with_context("test query")
            self.assertFalse(
                result["has_context"],
                "Should indicate no context when OpenAI is not available",
            )
            self.assertIn("Error", result["response"], "Response should indicate error")

    @mock.patch("RAGnificent.rag.pipeline.Pipeline.search_documents")
    def test_query_with_context_no_results(self, mock_search):
        """Test query_with_context when no search results are found."""
        mock_search.return_value = []

        result = self.pipeline.query_with_context("test query")
        self.assertFalse(
            result["has_context"], "Should indicate no context when no results found"
        )
        self.assertIn(
            "couldn't find",
            result["response"].lower(),
            "Response should indicate no information found",
        )

    @mock.patch("RAGnificent.rag.pipeline.openai")
    @mock.patch("RAGnificent.rag.pipeline.Pipeline.search_documents")
    def test_query_with_context_api_error(self, mock_search, mock_openai):
        """Test query_with_context when OpenAI API returns an error."""
        mock_search.return_value = [{"content": "Test content"}]

        mock_openai.chat.completions.create.side_effect = Exception("API error")

        result = self.pipeline.query_with_context("test query")
        self.assertTrue(result["has_context"], "Should indicate context is available")
        self.assertIn("Error", result["response"], "Response should indicate error")

    def test_get_pipeline_singleton(self):
        """Test that get_pipeline returns a singleton instance."""
        pipeline1 = get_pipeline()
        pipeline2 = get_pipeline()

        self.assertIs(
            pipeline1, pipeline2, "get_pipeline should return the same instance"
        )

        pipeline3 = get_pipeline(collection_name="different_collection")
        self.assertIs(
            pipeline1, pipeline3, "get_pipeline should still return the same instance"
        )


if __name__ == "__main__":
    unittest.main()
