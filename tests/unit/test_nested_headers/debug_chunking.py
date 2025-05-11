import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from RAGnificent.utils.chunk_utils import ContentChunker


def debug_chunking():
    chunker = ContentChunker(chunk_size=500, chunk_overlap=100)

    nested_markdown = """# Main Topic

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
The hierarchy should show that this is under Subtopic 2, not Subtopic 1.
"""

    sections = chunker._parse_markdown_sections(nested_markdown)
    for _i, _section in enumerate(sections):
        pass

    chunks = chunker.create_chunks_from_markdown(
        nested_markdown, "https://example.com/test"
    )

    for _i, chunk in enumerate(chunks):

        if "Nested Subtopic 1.1" in chunk.metadata.get("heading_path", ""):
            pass
        if "Nested Subtopic 2.1" in chunk.metadata.get("heading_path", ""):
            pass


if __name__ == "__main__":
    debug_chunking()
