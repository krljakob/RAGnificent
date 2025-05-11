import sys
import re
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def debug_markdown_parsing():
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
The hierarchy should show that this is under Subtopic 2, not Subtopic 1."""

    print("=== Markdown String Analysis ===")
    print(f"Type: {type(nested_markdown)}")
    print(f"Length: {len(nested_markdown)}")
    print(f"First 50 chars: {nested_markdown[:50]}")
    
    print("\n=== Line by Line Analysis ===")
    lines = nested_markdown.split('\n')
    print(f"Total lines: {len(lines)}")
    
    for i, line in enumerate(lines):
        print(f"\nLine {i+1}: '{line}'")
        
        is_header = False
        header_level = 0
        header_text = ""
        
        stripped_line = line.strip()
        
        match1 = re.match(r"^(#+)\s+(.*)", stripped_line)
        if match1:
            is_header = True
            header_level = len(match1[1])
            header_text = match1[2].strip()
            print(f"  MATCH 1: Level {header_level}, Text: '{header_text}'")
        
        match2 = re.match(r"^(#+)\s+(.*?)$", stripped_line)
        if match2:
            is_header = True
            header_level = len(match2[1])
            header_text = match2[2].strip()
            print(f"  MATCH 2: Level {header_level}, Text: '{header_text}'")
        
        if stripped_line and stripped_line[0] == '#':
            print(f"  MATCH 3: Line starts with #")
            
            count = 0
            for char in stripped_line:
                if char == '#':
                    count += 1
                else:
                    break
            print(f"  # Count: {count}")
        
        if is_header:
            print(f"  HEADER DETECTED: Level {header_level}, Text: '{header_text}'")
        else:
            print(f"  NOT A HEADER")

if __name__ == "__main__":
    debug_markdown_parsing()
