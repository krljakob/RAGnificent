import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def debug_string_literal():
    nested_markdown = r"""# Main Topic

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

    print("=== Raw String Analysis ===")
    print(f"Type: {type(nested_markdown)}")
    print(f"Length: {len(nested_markdown)}")
    
    print("\n=== Character by Character Analysis ===")
    for i, char in enumerate(nested_markdown[:100]):  # First 100 chars only
        print(f"Position {i}: '{char}' (ASCII: {ord(char)})")
    
    print("\n=== Line by Line with Explicit Newlines ===")
    lines = nested_markdown.split('\n')
    print(f"Total lines: {len(lines)}")
    
    for i, line in enumerate(lines):
        if i < 10:  # First 10 lines only
            print(f"Line {i+1}: {repr(line)}")
            
            if line.strip().startswith('#'):
                print(f"  HEADER: {line}")
                count = 0
                for char in line.strip():
                    if char == '#':
                        count += 1
                    else:
                        break
                print(f"  Level: {count}")

if __name__ == "__main__":
    debug_string_literal()
