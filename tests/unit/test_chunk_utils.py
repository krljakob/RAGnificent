import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Direct module import using absolute file paths
# Import fix applied
project_root = Path(__file__).parent.parent.parent
utils_path = project_root / "RAGnificent" / "utils"

# Add both the project root and the RAGnificent directory to sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "RAGnificent"))

# Import the needed module directly from its file path
chunk_utils_path = utils_path / "chunk_utils.py"
spec = importlib.util.spec_from_file_location("chunk_utils", chunk_utils_path)
chunk_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunk_utils)

# Extract the classes and functions we need
Chunk = chunk_utils.Chunk
ContentChunker = chunk_utils.ContentChunker
create_semantic_chunks = chunk_utils.create_semantic_chunks


class TestChunkUtils(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.chunker = ContentChunker()
        self.test_url = "https://example.com/test"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_create_chunks_from_markdown(self):
        """Test chunk creation from markdown content."""
        # Test markdown with multiple sections
        test_markdown = """# Title

## Section 1
This is content for section 1.
It has multiple lines.

## Section 2
This is content for section 2.
It also has multiple lines.
"""

        chunks = self.chunker.create_chunks_from_markdown(test_markdown, self.test_url)

        # Verify chunk count and structure
        self.assertGreater(len(chunks), 0, "Should create at least one chunk")
        self.assertLessEqual(
            len(chunks), 5, "Should not create excessive chunks for simple content"
        )

        # Verify actual chunk count matches content structure
        # Count actual markdown headers (lines starting with #)
        lines = test_markdown.strip().split("\n")
        header_lines = [line for line in lines if line.strip().startswith("#")]
        expected_sections = len(header_lines)
        self.assertEqual(
            len(chunks),
            expected_sections,
            f"Should create {expected_sections} chunks for {expected_sections} sections",
        )

        # Verify chunk content contains expected headers
        chunk_contents = [chunk.content for chunk in chunks]
        self.assertIn(
            "# Title", chunk_contents[0], "First chunk should contain main title"
        )

        # Verify all sections are represented in chunks
        section_headers = ["# Title", "## Section 1", "## Section 2"]
        for header in section_headers:
            found_in_chunk = any(header in content for content in chunk_contents)
            self.assertTrue(
                found_in_chunk,
                f"Header '{header}' should be found in at least one chunk",
            )

        # Verify metadata structure and completeness
        for i, chunk in enumerate(chunks):
            self.assertIsNotNone(chunk.metadata, f"Chunk {i} should have metadata")
            self.assertIn(
                "heading", chunk.metadata, f"Chunk {i} should have heading metadata"
            )
            self.assertNotEqual(
                chunk.metadata["heading"], "", f"Chunk {i} heading should not be empty"
            )
            self.assertEqual(
                chunk.source_url,
                self.test_url,
                f"Chunk {i} should have correct source URL",
            )
            self.assertEqual(
                chunk.chunk_type, "section", f"Chunk {i} should be of type 'section'"
            )
            self.assertIsNotNone(chunk.id, f"Chunk {i} should have an ID")
            self.assertNotEqual(chunk.id, "", f"Chunk {i} ID should not be empty")

    def test_create_large_chunks(self):
        """Test handling of sections larger than chunk_size."""
        # Create a very large section
        large_section = "# Large Section\n" + "This is a word. " * 500

        # Use a smaller chunk size to force chunking
        small_chunker = ContentChunker(chunk_size=100, chunk_overlap=20)
        chunks = small_chunker.create_chunks_from_markdown(large_section, self.test_url)

        # Verify chunking behavior for oversized content
        self.assertGreater(
            len(chunks), 1, "Large content should be split into multiple chunks"
        )

        # Verify chunk size constraints are respected
        for i, chunk in enumerate(chunks):
            self.assertLessEqual(
                len(chunk.content),
                small_chunker.chunk_size + 200,  # Allow some tolerance
                f"Chunk {i} content length {len(chunk.content)} exceeds reasonable limit",
            )

        # Verify metadata consistency across chunks from same section
        for chunk in chunks:
            self.assertEqual(
                chunk.metadata["heading"],
                "# Large Section",
                "All chunks from same section should have same heading",
            )
            self.assertEqual(
                chunk.source_url,
                self.test_url,
                "All chunks should have correct source URL",
            )

        # Verify content continuity (no content is lost)
        combined_content = " ".join(chunk.content for chunk in chunks)
        self.assertIn(
            "Large Section", combined_content, "Header should be preserved in chunks"
        )
        word_count_original = len(large_section.split())
        word_count_chunks = len(combined_content.split())
        # Allow for some duplication due to overlap but ensure most content is preserved
        self.assertGreater(
            word_count_chunks,
            word_count_original * 0.8,
            "Most original content should be preserved in chunks",
        )

    def test_save_chunks_jsonl(self):
        """Test saving chunks to JSONL format."""
        chunks = [
            Chunk(
                id="123",
                content="Test content 1",
                metadata={"heading": "Test", "domain": "example.com"},
                source_url=self.test_url,
                created_at="2023-01-01T00:00:00",
                chunk_type="section",
            ),
            Chunk(
                id="456",
                content="Test content 2",
                metadata={"heading": "Test 2", "domain": "example.com"},
                source_url=self.test_url,
                created_at="2023-01-01T00:00:00",
                chunk_type="section",
            ),
        ]

        self.chunker.save_chunks(chunks, self.test_dir, "jsonl")

        jsonl_path = Path(self.test_dir) / "chunks.jsonl"
        self.assertTrue(jsonl_path.exists())

        # Verify file contents and structure
        with open(jsonl_path, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(chunks), "Should write one line per chunk")

            for i, line in enumerate(lines):
                self.assertNotEqual(line.strip(), "", f"Line {i} should not be empty")

                try:
                    chunk_data = json.loads(line)
                except json.JSONDecodeError as e:
                    self.fail(f"Line {i} is not valid JSON: {e}")

                # Verify required fields are present
                required_fields = [
                    "id",
                    "content",
                    "metadata",
                    "source_url",
                    "chunk_type",
                ]
                for field in required_fields:
                    self.assertIn(
                        field, chunk_data, f"Chunk {i} missing required field: {field}"
                    )

                # Verify field values match original chunks
                original_chunk = chunks[i]
                self.assertEqual(
                    chunk_data["id"], original_chunk.id, f"Chunk {i} ID mismatch"
                )
                self.assertEqual(
                    chunk_data["content"],
                    original_chunk.content,
                    f"Chunk {i} content mismatch",
                )
                self.assertEqual(
                    chunk_data["source_url"],
                    original_chunk.source_url,
                    f"Chunk {i} source_url mismatch",
                )
                self.assertEqual(
                    chunk_data["chunk_type"],
                    original_chunk.chunk_type,
                    f"Chunk {i} chunk_type mismatch",
                )

    def test_create_semantic_chunks(self):
        """Test the create_semantic_chunks convenience function."""
        # Test with markdown content
        markdown_content = "# Test Header\n\nThis is test content with a header."
        markdown_chunks = create_semantic_chunks(markdown_content, self.test_url)

        self.assertGreater(
            len(markdown_chunks), 0, "Should create at least one chunk from markdown"
        )
        self.assertIsNotNone(
            markdown_chunks[0].metadata, "Markdown chunk should have metadata"
        )
        self.assertIn(
            "heading",
            markdown_chunks[0].metadata,
            "Markdown chunk should have heading metadata",
        )
        self.assertEqual(
            markdown_chunks[0].source_url,
            self.test_url,
            "Chunk should have correct source URL",
        )

        # Test with plain text content
        text_content = "This is just plain text without any markdown headers or special formatting."
        text_chunks = create_semantic_chunks(text_content, self.test_url)

        self.assertEqual(
            len(text_chunks), 1, "Plain text should create exactly one chunk"
        )
        self.assertEqual(
            text_chunks[0].chunk_type,
            "text_chunk",
            "Plain text should create text_chunk type",
        )
        self.assertEqual(
            text_chunks[0].content,
            text_content,
            "Plain text content should be preserved exactly",
        )
        self.assertEqual(
            text_chunks[0].source_url,
            self.test_url,
            "Plain text chunk should have correct source URL",
        )

        # Test with empty content
        empty_chunks = create_semantic_chunks("", self.test_url)
        self.assertEqual(len(empty_chunks), 0, "Empty content should create no chunks")

        # Test with whitespace-only content
        whitespace_chunks = create_semantic_chunks("   \n\t\n   ", self.test_url)
        self.assertEqual(
            len(whitespace_chunks), 0, "Whitespace-only content should create no chunks"
        )


if __name__ == "__main__":
    unittest.main()
