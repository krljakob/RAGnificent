import sys
import re
from pathlib import Path
from pprint import pprint

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from RAGnificent.utils.chunk_utils import ContentChunker

def debug_header_parsing():
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

    print("=== Raw Markdown Lines ===")
    for i, line in enumerate(nested_markdown.split('\n')):
        print(f"Line {i+1}: '{line}'")
    
    print("\n=== Testing Header Regex ===")
    for i, line in enumerate(nested_markdown.split('\n')):
        match = re.match(r"^(#+)\s+(.*?)$", line.strip())
        if match:
            print(f"Line {i+1} matched: '{line}'")
            print(f"  Level: {len(match[1])}")
            print(f"  Text: '{match[2].strip()}'")
        else:
            if line.strip().startswith('#'):
                print(f"Line {i+1} FAILED to match but starts with #: '{line}'")
    
    chunker = ContentChunker(chunk_size=500, chunk_overlap=100)
    
    print("\n=== Parsing Markdown Sections ===")
    sections = chunker._parse_markdown_sections(nested_markdown)
    print(f"Total sections found: {len(sections)}")
    
    for i, section in enumerate(sections):
        print(f"\nSection {i+1}:")
        print(f"  Heading: {section['heading']}")
        print(f"  Path: {section['path']}")
        print(f"  Level: {section['level']}")
        print(f"  Path Elements: {section.get('path_elements', 'N/A')}")
        print(f"  Parent Headers: {section.get('parent_headers', 'N/A')}")
        print(f"  Content Length: {len(section['content'])}")
        print(f"  Content Preview: {section['content'][:50]}...")

if __name__ == "__main__":
    debug_header_parsing()
