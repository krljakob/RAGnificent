import re
import sys
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

    lines = nested_markdown.split("\n")

    for _i, line in enumerate(lines):

        is_header = False

        stripped_line = line.strip()

        if match1 := re.match(r"^(#+)\s+(.*)", stripped_line):
            is_header = True
            len(match1[1])
            match1[2].strip()

        if match2 := re.match(r"^(#+)\s+(.*?)$", stripped_line):
            is_header = True
            len(match2[1])
            match2[2].strip()

        if stripped_line and stripped_line[0] == "#":

            count = 0
            for char in stripped_line:
                if char == "#":
                    count += 1
                else:
                    break


if __name__ == "__main__":
    debug_markdown_parsing()
