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

        self.nested_markdown = """# Main Topic

This is an introduction to the main topic.


This is content under subtopic 1.


This is deeply nested content with a lot of text that should be split into multiple chunks.
This paragraph is intentionally long to ensure that the chunking algorithm splits it.
We want to test that the nested header structure is preserved when chunking occurs.
The parent headers should be included in the metadata and in the content of continuation chunks.
This ensures that when the chunks are used in a RAG system, the context is preserved.


This is content under subtopic 2, which is at the same level as subtopic 1 but comes after the nested content.


More nested content here that will also need to be split into multiple chunks.
Again, this paragraph is intentionally verbose to trigger the chunking algorithm.
We want to verify that the parent headers are correctly tracked and included.
The hierarchy should show that this is under Subtopic 2, not Subtopic 1."""

    def test_nested_header_chunking(self):
        """Test that nested headers are properly handled in chunking."""
        chunks = self.chunker.create_chunks_from_markdown(
            self.nested_markdown, "https://example.com/test"
        )

        for _i, chunk in enumerate(chunks):
            pass

        self.assertGreater(len(chunks), 1, "Should create multiple chunks")

        subtopic_1_1_chunks = [
            chunk
            for chunk in chunks
            if "Nested Subtopic 1.1" in chunk.metadata.get("heading_path", "")
        ]

        self.assertGreater(
            len(subtopic_1_1_chunks), 0, "Should have chunks for Nested Subtopic 1.1"
        )

        if len(subtopic_1_1_chunks) > 1:
            continuation_chunks = [
                chunk
                for chunk in subtopic_1_1_chunks
                if chunk.metadata.get("is_continuation", False)
            ]

            for chunk in continuation_chunks:
                self.assertIn("parent_headers", chunk.metadata)
                parent_headers = chunk.metadata["parent_headers"]

                header_texts = [h["text"] for h in parent_headers]
                self.assertIn("Main Topic", header_texts)
                self.assertIn("Subtopic 1", header_texts)

                self.assertIn("# Main Topic", chunk.content)
                self.assertIn("## Subtopic 1", chunk.content)
                self.assertIn("### Nested Subtopic 1.1", chunk.content)

        if subtopic_2_1_chunks := [
            chunk
            for chunk in chunks
            if "Nested Subtopic 2.1" in chunk.metadata.get("heading_path", "")
        ]:
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

        if nested_chunks := [
            chunk for chunk in chunks if chunk.metadata.get("nested_level", 0) > 1
        ]:
            chunk = nested_chunks[0]

            self.assertIsInstance(chunk.metadata["path_elements"], list)

            path_string = " > ".join(chunk.metadata["path_elements"])
            self.assertEqual(path_string, chunk.metadata["heading_path"])


if __name__ == "__main__":
    unittest.main()
