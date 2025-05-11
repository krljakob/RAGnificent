import sys
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
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

        self.assertEqual(len(sections), 5, "Should find exactly 5 sections")

        header_levels = [section["level"] for section in sections]
        self.assertEqual(
            header_levels,
            [1, 2, 3, 2, 3],
            "Header levels should match expected hierarchy",
        )

        paths = [section["path"] for section in sections]
        self.assertEqual(paths[0], "Main Topic")
        self.assertEqual(paths[1], "Main Topic > Subtopic 1")
        self.assertEqual(paths[2], "Main Topic > Subtopic 1 > Nested Subtopic 1.1")
        self.assertEqual(paths[3], "Main Topic > Subtopic 2")
        self.assertEqual(paths[4], "Main Topic > Subtopic 2 > Nested Subtopic 2.1")

        nested_levels = [section.get("nested_level", -1) for section in sections]
        self.assertEqual(
            nested_levels,
            [0, 1, 2, 1, 2],
            "Nested levels should match expected hierarchy",
        )

        self.assertEqual(
            len(sections[0]["parent_headers"]),
            0,
            "Main topic should have no parent headers",
        )
        self.assertEqual(
            len(sections[1]["parent_headers"]),
            1,
            "Subtopic 1 should have 1 parent header",
        )
        self.assertEqual(
            len(sections[2]["parent_headers"]),
            2,
            "Nested Subtopic 1.1 should have 2 parent headers",
        )
        self.assertEqual(
            len(sections[3]["parent_headers"]),
            1,
            "Subtopic 2 should have 1 parent header",
        )
        self.assertEqual(
            len(sections[4]["parent_headers"]),
            2,
            "Nested Subtopic 2.1 should have 2 parent headers",
        )

        self.assertEqual(
            sections[2]["parent_headers"][0]["text"],
            "Main Topic",
            "First parent of Nested Subtopic 1.1 should be Main Topic",
        )
        self.assertEqual(
            sections[2]["parent_headers"][1]["text"],
            "Subtopic 1",
            "Second parent of Nested Subtopic 1.1 should be Subtopic 1",
        )
        self.assertEqual(
            sections[4]["parent_headers"][0]["text"],
            "Main Topic",
            "First parent of Nested Subtopic 2.1 should be Main Topic",
        )
        self.assertEqual(
            sections[4]["parent_headers"][1]["text"],
            "Subtopic 2",
            "Second parent of Nested Subtopic 2.1 should be Subtopic 2",
        )

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

        subtopic_2_1_chunks = [
            chunk
            for chunk in chunks
            if "Nested Subtopic 2.1" in chunk.metadata.get("heading_path", "")
        ]

        self.assertGreater(
            len(subtopic_2_1_chunks), 0, "Should have chunks for Nested Subtopic 2.1"
        )

        for chunk in subtopic_2_1_chunks:
            parent_headers = chunk.metadata["parent_headers"]
            header_texts = [h["text"] for h in parent_headers]

            self.assertIn("Main Topic", header_texts)
            self.assertIn("Subtopic 2", header_texts)
            self.assertNotIn("Subtopic 1", header_texts)

            self.assertEqual(chunk.metadata["nested_level"], 2)

    def test_path_elements_structure(self):
        """Test that path_elements correctly represents the header hierarchy."""
        chunks = self.chunker.create_chunks_from_markdown(
            self.nested_markdown, "https://example.com/test"
        )

        nested_chunks = [
            chunk for chunk in chunks if chunk.metadata.get("nested_level", 0) > 0
        ]

        self.assertGreater(
            len(nested_chunks), 0, "Should have chunks with nested headers"
        )

        for chunk in nested_chunks:
            self.assertIsInstance(chunk.metadata["path_elements"], list)

            path_string = " > ".join(chunk.metadata["path_elements"])
            self.assertEqual(path_string, chunk.metadata["heading_path"])

            self.assertEqual(
                chunk.metadata["nested_level"],
                len(chunk.metadata["parent_headers"]),
                "Nested level should match number of parent headers",
            )


if __name__ == "__main__":
    unittest.main()
