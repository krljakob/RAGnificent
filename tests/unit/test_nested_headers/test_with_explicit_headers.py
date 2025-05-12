import sys
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from RAGnificent.utils.chunk_utils import ContentChunker


class TestNestedHeaderChunking(unittest.TestCase):
    """Test the improved nested header handling in the chunking algorithm."""

    def setUp(self):
        """Set up test environment."""
        self.chunker = ContentChunker(chunk_size=500, chunk_overlap=100)

        self.nested_markdown = "# Main Topic\n\nThis is an introduction to the main topic.\n\n## Subtopic 1\n\nThis is content under subtopic 1.\n\n### Nested Subtopic 1.1\n\nThis is deeply nested content with a lot of text that should be split into multiple chunks.\nThis paragraph is intentionally long to ensure that the chunking algorithm splits it.\nWe want to test that the nested header structure is preserved when chunking occurs.\nThe parent headers should be included in the metadata and in the content of continuation chunks.\nThis ensures that when the chunks are used in a RAG system, the context is preserved.\n\n## Subtopic 2\n\nThis is content under subtopic 2, which is at the same level as subtopic 1 but comes after the nested content.\n\n### Nested Subtopic 2.1\n\nMore nested content here that will also need to be split into multiple chunks.\nAgain, this paragraph is intentionally verbose to trigger the chunking algorithm.\nWe want to verify that the parent headers are correctly tracked and included.\nThe hierarchy should show that this is under Subtopic 2, not Subtopic 1."

    def test_section_parsing(self):
        """Test that markdown sections are correctly parsed."""
        sections = self.chunker._parse_markdown_sections(self.nested_markdown)

        self.assertGreaterEqual(len(sections), 5, "Should find at least 5 sections")

        header_levels = [
            section["level"] for section in sections if section["level"] > 0
        ]
        self.assertEqual(
            header_levels,
            [1, 2, 3, 2, 3],
            "Header levels should match expected hierarchy",
        )

        paths = [section["path"] for section in sections if section["level"] > 0]
        self.assertIn("Main Topic", paths[0])
        self.assertIn("Main Topic > Subtopic 1", paths[1])
        self.assertIn("Main Topic > Subtopic 1 > Nested Subtopic 1.1", paths[2])
        self.assertIn("Main Topic > Subtopic 2", paths[3])
        self.assertIn("Main Topic > Subtopic 2 > Nested Subtopic 2.1", paths[4])

    def test_nested_header_chunking(self):
        """Test that nested headers are properly handled in chunking."""
        chunks = self.chunker.create_chunks_from_markdown(
            self.nested_markdown, "https://example.com/test"
        )

        self.assertGreater(len(chunks), 1, "Should create multiple chunks")

        subtopic_1_1_chunks = [
            chunk
            for chunk in chunks
            if "Nested Subtopic 1.1" in chunk.metadata.get("heading_path", "")
        ]

        self.assertGreater(
            len(subtopic_1_1_chunks), 0, "Should have chunks for Nested Subtopic 1.1"
        )


if __name__ == "__main__":
    unittest.main()
