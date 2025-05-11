import sys
from pathlib import Path
from pprint import pprint

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
    
    print("=== Parsing Markdown Sections ===")
    sections = chunker._parse_markdown_sections(nested_markdown)
    for i, section in enumerate(sections):
        print(f"\nSection {i+1}:")
        print(f"  Heading: {section['heading']}")
        print(f"  Path: {section['path']}")
        print(f"  Level: {section['level']}")
        print(f"  Path Elements: {section.get('path_elements', 'N/A')}")
        print(f"  Parent Headers: {section.get('parent_headers', 'N/A')}")
    
    print("\n=== Creating Chunks ===")
    chunks = chunker.create_chunks_from_markdown(nested_markdown, "https://example.com/test")
    print(f"Total chunks created: {len(chunks)}")
    
    print("\n=== Chunk Metadata ===")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Heading Path: {chunk.metadata.get('heading_path', 'N/A')}")
        print(f"  Nested Level: {chunk.metadata.get('nested_level', 'N/A')}")
        print(f"  Is Continuation: {chunk.metadata.get('is_continuation', 'N/A')}")
        
        if "Nested Subtopic 1.1" in chunk.metadata.get('heading_path', ''):
            print("  ** Contains Nested Subtopic 1.1 **")
        if "Nested Subtopic 2.1" in chunk.metadata.get('heading_path', ''):
            print("  ** Contains Nested Subtopic 2.1 **")

if __name__ == "__main__":
    debug_chunking()
