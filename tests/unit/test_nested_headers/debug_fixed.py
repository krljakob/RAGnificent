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
        original_match = re.match(r"^(#+)\s+(.*)", line)
        modified_match = re.match(r"^(#+)\s+(.*)", line.strip())
        
        if original_match:
            print(f"Line {i+1} matched original regex: '{line}'")
            print(f"  Level: {len(original_match[1])}")
            print(f"  Text: '{original_match[2].strip()}'")
        elif modified_match:
            print(f"Line {i+1} matched ONLY modified regex: '{line}'")
            print(f"  Level: {len(modified_match[1])}")
            print(f"  Text: '{modified_match[2].strip()}'")
        else:
            if line.strip().startswith('#'):
                print(f"Line {i+1} FAILED to match but starts with #: '{line}'")
    
    def parse_markdown_sections_fixed(markdown_content):
        sections = []
        lines = markdown_content.split("\n")
        
        header_stack = []
        current_section = None
        
        for line in lines:
            if header_match := re.match(r"^(#+)\s+(.*)", line.strip()):
                level = len(header_match[1])
                heading_text = header_match[2].strip()
                
                if current_section:
                    sections.append(current_section)
                
                while header_stack and header_stack[-1]["level"] >= level:
                    header_stack.pop()
                
                path_elements = [h["text"] for h in header_stack] + [heading_text]
                path = " > ".join(path_elements)
                
                parent_headers = []
                for header in header_stack:
                    parent_headers.append({
                        "text": header["text"],
                        "level": header["level"],
                        "markdown": "#" * header["level"] + " " + header["text"]
                    })
                
                header_entry = {"level": level, "text": heading_text}
                
                header_stack.append(header_entry)
                
                current_section = {
                    "heading": line.strip(),
                    "content": line.strip() + "\n",
                    "level": level,
                    "path": path,
                    "path_elements": path_elements,
                    "parent_headers": parent_headers,
                }
            elif current_section:
                current_section["content"] += line + "\n"
            elif line.strip():
                current_section = {
                    "heading": "Document Start",
                    "content": line + "\n",
                    "level": 0,
                    "path": "Document Start",
                    "path_elements": ["Document Start"],
                    "parent_headers": [],
                }
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    print("\n=== Parsing Markdown Sections (Fixed) ===")
    sections = parse_markdown_sections_fixed(nested_markdown)
    print(f"Total sections found: {len(sections)}")
    
    for i, section in enumerate(sections):
        print(f"\nSection {i+1}:")
        print(f"  Heading: {section['heading']}")
        print(f"  Path: {section['path']}")
        print(f"  Level: {section['level']}")
        print(f"  Path Elements: {section.get('path_elements', 'N/A')}")
        print(f"  Parent Headers Count: {len(section.get('parent_headers', []))}")
        if section.get('parent_headers'):
            for j, ph in enumerate(section['parent_headers']):
                print(f"    Parent {j+1}: {ph['text']} (Level {ph['level']})")
        print(f"  Content Length: {len(section['content'])}")
        print(f"  Content Preview: {section['content'][:50]}...")

if __name__ == "__main__":
    debug_header_parsing()
