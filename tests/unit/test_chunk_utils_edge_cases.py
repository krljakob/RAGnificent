"""
Test edge cases in the chunk_utils module.
"""

import random
import shutil
import string
import tempfile
import unittest
from pathlib import Path

from RAGnificent.utils.chunk_utils import ContentChunker, create_semantic_chunks


class TestChunkingEdgeCases(unittest.TestCase):
    """Test edge cases for the chunking utilities."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_empty_content(self):
        """Test handling of empty content."""
        chunks = create_semantic_chunks("", "https://example.com")
        self.assertEqual(len(chunks), 0, "Empty content should yield no chunks")

    def test_whitespace_only_content(self):
        """Test handling of whitespace-only content."""
        chunks = create_semantic_chunks("   \n   \t   ", "https://example.com")
        self.assertEqual(
            len(chunks), 0, "Whitespace-only content should yield no chunks"
        )

    def test_very_small_content(self):
        """Test handling of very small content."""
        test_content = "This is a small text snippet."
        chunks = create_semantic_chunks(test_content, "https://example.com")
        self.assertEqual(len(chunks), 1, "Small content should yield exactly one chunk")
        self.assertEqual(chunks[0].content, test_content)

    def test_nested_headers(self):
        """Test proper handling of nested headers in markdown."""
        markdown_content = """# Top Level Header
Some text at the top level.

## Second Level Header 1
Content under second level 1.

### Third Level Header A
Deep nested content A.

### Third Level Header B
Deep nested content B.

## Second Level Header 2
Content under second level 2.
"""
        chunks = create_semantic_chunks(markdown_content, "https://example.com")

        # Check that the hierarchy is properly represented
        paths = [chunk.metadata.get("heading_path", "") for chunk in chunks]

        # Verify we have the correct paths
        self.assertTrue(any("Top Level Header" in path for path in paths))
        self.assertTrue(
            any("Top Level Header > Second Level Header 1" in path for path in paths)
        )
        self.assertTrue(
            any(
                "Top Level Header > Second Level Header 1 > Third Level Header A"
                in path
                for path in paths
            )
        )
        self.assertTrue(
            any(
                "Top Level Header > Second Level Header 1 > Third Level Header B"
                in path
                for path in paths
            )
        )
        self.assertTrue(
            any("Top Level Header > Second Level Header 2" in path for path in paths)
        )

    def test_very_long_header(self):
        """Test handling of extremely long headers."""
        # Create a very long header
        long_header = "# " + "".join(random.choices(string.ascii_letters, k=500))
        content = f"{long_header}\nSome content under a very long header."

        chunks = create_semantic_chunks(content, "https://example.com")
        self.assertTrue(len(chunks) > 0, "Should handle very long headers")

    def test_malformed_headers(self):
        """Test handling of malformed headers."""
        malformed_content = """
#Malformed header (no space)
Content under malformed header.

##  Multiple spaces
Content with extra spaces.

###### Too many levels
Very deep header.
"""
        chunks = create_semantic_chunks(malformed_content, "https://example.com")
        # Should still create chunks even with malformed headers
        self.assertTrue(len(chunks) > 0)

    def test_very_large_document(self):
        """Test handling of extremely large documents."""
        # Generate a large markdown document
        large_doc = []
        for i in range(50):  # 50 sections
            large_doc.append(f"## Section {i}")
            # Add 20 paragraphs per section
            large_doc.extend(
                f"This is paragraph {j} in section {i}. " * 10 for j in range(20)
            )
            large_doc.append("")  # Empty line between sections

        large_content = "\n".join(large_doc)

        # Should handle without memory issues
        chunks = create_semantic_chunks(large_content, "https://example.com")
        self.assertTrue(
            len(chunks) > 50, "Should create many chunks for large document"
        )

    def test_special_characters(self):
        """Test handling of special characters in markdown."""
        special_content = """# Header with Spéçial Chàrácters!
Text with emoji 🔄📝 and unicode ™ ® ©.

## Header with code `inline code` elements
```python
# Code block with special characters
def special_func(x):
    return f"Result: {x * 2}"
```

## Math equations: $E = mc^2$

* List item 1
* List item with **bold** and *italic*
"""
        chunks = create_semantic_chunks(special_content, "https://example.com")
        self.assertTrue(len(chunks) > 0, "Should handle special characters properly")

    def test_missing_source_url(self):
        """Test handling of missing source URL."""
        with self.assertRaises(Exception):
            create_semantic_chunks("Some content", None)

    def test_chunk_overlap_larger_than_chunk_size(self):
        """Test handling of invalid chunking parameters."""
        # Create a chunker with overlap larger than chunk size
        chunker = ContentChunker(chunk_size=100, chunk_overlap=150)

        # Should still work, but overlap will effectively be reduced
        chunks = chunker.create_chunks_from_markdown(
            "# Header\nSome content.\n\n## Another header\nMore content.",
            "https://example.com",
        )
        self.assertTrue(
            len(chunks) > 0, "Should handle invalid chunking parameters gracefully"
        )

    def test_jsonl_save_format(self):
        """Test saving to JSONL format."""
        chunks = create_semantic_chunks("# Test\nSome content.", "https://example.com")
        self.assertTrue(len(chunks) > 0)

        # Save to JSONL format
        chunker = ContentChunker()
        output_dir = Path(self.temp_dir) / "jsonl_chunks"
        chunker.save_chunks(chunks, str(output_dir), "jsonl")

        # Verify the output file exists
        output_file = output_dir / "chunks.jsonl"
        self.assertTrue(output_file.exists(), "JSONL output file should be created")

    def test_json_save_format(self):
        """Test saving to individual JSON files."""
        chunks = create_semantic_chunks("# Test\nSome content.", "https://example.com")
        self.assertTrue(len(chunks) > 0)

        # Save to JSON format
        chunker = ContentChunker()
        output_dir = Path(self.temp_dir) / "json_chunks"
        chunker.save_chunks(chunks, str(output_dir), "json")

        # Verify the output files exist
        for chunk in chunks:
            output_file = output_dir / f"{chunk.id}.json"
            self.assertTrue(
                output_file.exists(),
                f"JSON output file for chunk {chunk.id} should be created",
            )

    def test_identical_chunks_get_unique_ids(self):
        """Test that identical chunks from different source URLs get unique IDs."""
        content = "# Test Header\nSome identical content."

        chunks1 = create_semantic_chunks(content, "https://example.com/page1")
        chunks2 = create_semantic_chunks(content, "https://example.com/page2")

        # Check that the chunks have different IDs
        self.assertNotEqual(
            chunks1[0].id,
            chunks2[0].id,
            "Identical chunks from different sources should have unique IDs",
        )


if __name__ == "__main__":
    unittest.main()
