"""
Utility module for chunking text content for RAG (Retrieval Augmented Generation).
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse


@dataclass
class Chunk:
    """Represents a chunk of content for RAG processing."""

    id: str
    content: str
    metadata: Dict[str, Any]
    source_url: str
    created_at: str
    chunk_type: str


class ContentChunker:
    """Handles chunking of content for RAG systems."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum size of content chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks_from_markdown(
        self, markdown_content: str, source_url: str
    ) -> List[Chunk]:
        """
        Split the markdown content into chunks for RAG processing, preserving header hierarchy.

        Args:
            markdown_content: The markdown content to chunk
            source_url: The URL the content was scraped from

        Returns:
            A list of Chunk objects
        """
        # Parse hierarchy of sections using header levels
        sections = self._parse_markdown_sections(markdown_content)

        # Now create chunks from hierarchical sections
        chunks = []

        # Parse domain for metadata
        domain = urlparse(source_url).netloc

        for section in sections:
            heading = section["heading"]
            content = section["content"]
            heading_level = section["level"]
            heading_path = section["path"]
            parent_headers = section.get("parent_headers", [])
            path_elements = section.get("path_elements", [])

            # If the section is smaller than chunk_size, keep it as one chunk
            if len(content) <= self.chunk_size:
                chunk_id = hashlib.md5(
                    f"{source_url}:{heading_path}".encode()
                ).hexdigest()
                chunk = Chunk(
                    id=chunk_id,
                    content=content,
                    metadata={
                        "heading": heading,
                        "heading_level": heading_level,
                        "heading_path": heading_path,
                        "path_elements": path_elements,
                        "parent_headers": parent_headers,
                        "domain": domain,
                        "word_count": len(content.split()),
                        "char_count": len(content),
                        "nested_level": len(parent_headers),
                    },
                    source_url=source_url,
                    created_at=datetime.now().isoformat(),
                    chunk_type="section",
                )
                chunks.append(chunk)
            else:
                # Split into overlapping chunks
                words = content.split()
                words_per_chunk = (
                    self.chunk_size // 5
                )  # Approximate words per character
                overlap_words = self.chunk_overlap // 5

                for i in range(0, len(words), words_per_chunk - overlap_words):
                    chunk_words = words[i : i + words_per_chunk]
                    if not chunk_words:
                        continue

                    if i > 0:
                        context_headers = [
                            parent["markdown"] for parent in parent_headers
                        ]
                        # Add the current section header
                        if heading and not " ".join(chunk_words).startswith(heading):
                            context_headers.append(heading)

                        if context_headers:
                            context_prefix = "\n".join(context_headers) + "\n\n"
                            chunk_content = context_prefix + " ".join(chunk_words)
                        else:
                            chunk_content = " ".join(chunk_words)
                    else:
                        chunk_content = " ".join(chunk_words)

                    chunk_id = hashlib.md5(
                        f"{source_url}:{heading_path}:{i}".encode()
                    ).hexdigest()

                    chunk = Chunk(
                        id=chunk_id,
                        content=chunk_content,
                        metadata={
                            "heading": heading,
                            "heading_level": heading_level,
                            "heading_path": heading_path,
                            "path_elements": path_elements,
                            "parent_headers": parent_headers,
                            "domain": domain,
                            "position": i // (words_per_chunk - overlap_words),
                            "word_count": len(chunk_words),
                            "char_count": len(chunk_content),
                            "nested_level": len(parent_headers),
                            "is_continuation": i > 0,
                        },
                        source_url=source_url,
                        created_at=datetime.now().isoformat(),
                        chunk_type="content_chunk",
                    )
                    chunks.append(chunk)

        return chunks

    def _parse_markdown_sections(self, markdown_content: str) -> List[Dict[str, Any]]:
        """
        Parse markdown into hierarchical sections based on headers.

        Args:
            markdown_content: The markdown content to parse

        Returns:
            A list of section dictionaries containing heading, content, level, path, and parent_headers
        """
        sections = []
        lines = markdown_content.split("\n")

        # Track header stack for maintaining hierarchy
        header_stack = []
        current_section = None

        for line in lines:
            # Check if line is a header by matching up to three leading spaces followed by #
            if header_match := re.match(r"^ {0,3}(#+)\s+(.*?)$", line):
                # This is a header line
                level = len(header_match[1])
                heading_text = header_match[2].strip()

                # If we have a current section, finalize it and add to sections list
                if current_section:
                    sections.append(current_section)

                # Update header stack based on new header level
                while header_stack and header_stack[-1]["level"] >= level:
                    header_stack.pop()

                # Create path representation of current location in hierarchy
                path_elements = [h["text"] for h in header_stack] + [heading_text]
                path = " > ".join(path_elements)

                parent_headers = [
                    {
                        "text": header["text"],
                        "level": header["level"],
                        "markdown": "#" * header["level"] + " " + header["text"],
                    }
                    for header in header_stack
                ]
                # Create new header entry
                header_entry = {"level": level, "text": heading_text}

                # Push to stack
                header_stack.append(header_entry)

                # Start new section
                current_section = {
                    "heading": line,
                    "content": line + "\n",
                    "level": level,
                    "path": path,
                    "path_elements": path_elements,
                    "parent_headers": parent_headers,
                    "nested_level": len(parent_headers),
                }
            elif current_section:
                # Add line to current section content
                current_section["content"] += line + "\n"
            elif line.strip():
                # Create a default section for content before first header
                current_section = {
                    "heading": "Document Start",
                    "content": line + "\n",
                    "level": 0,
                    "path": "Document Start",
                    "path_elements": ["Document Start"],
                    "parent_headers": [],
                    "nested_level": 0,
                }

        # Add the final section if it exists
        if current_section:
            sections.append(current_section)

        return sections

    def save_chunks(
        self, chunks: List[Chunk], output_dir: str, output_format: str = "jsonl"
    ) -> None:
        """
        Save content chunks to files for RAG processing.

        Args:
            chunks: List of Chunk objects to save
            output_dir: Directory to save the chunks
            output_format: Format to save chunks in (json or jsonl)
        """
        # Create the output directory if it doesn't exist
        chunk_dir = Path(output_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "jsonl":
            # Save all chunks to a single JSONL file
            output_file = chunk_dir / "chunks.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(asdict(chunk)) + "\n")
            return
        # Save each chunk as a separate JSON file
        for chunk in chunks:
            output_file = chunk_dir / f"{chunk.id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(asdict(chunk), f, indent=2)


def create_semantic_chunks(
    content: str, source_url: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Chunk]:
    """
    Convenience function to create semantic chunks from any content.

    Args:
        content: The content to chunk
        source_url: The source URL of the content
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        A list of Chunk objects
    """
    chunker = ContentChunker(chunk_size, chunk_overlap)

    # Check if content is likely markdown
    if re.search(r"^#+ ", content, re.MULTILINE):
        return chunker.create_chunks_from_markdown(content, source_url)

    # For non-markdown text, create simple overlapping chunks
    chunks = []
    domain = urlparse(source_url).netloc
    words = content.split()
    words_per_chunk = chunk_size // 5  # Approximate words per character
    overlap_words = chunk_overlap // 5

    for i in range(0, len(words), words_per_chunk - overlap_words):
        chunk_words = words[i : i + words_per_chunk]
        if not chunk_words:
            continue

        chunk_content = " ".join(chunk_words)
        chunk_id = hashlib.md5(f"{source_url}:text:{i}".encode()).hexdigest()

        chunk = Chunk(
            id=chunk_id,
            content=chunk_content,
            metadata={
                "domain": domain,
                "position": i // (words_per_chunk - overlap_words),
                "word_count": len(chunk_words),
                "char_count": len(chunk_content),
            },
            source_url=source_url,
            created_at=datetime.now().isoformat(),
            chunk_type="text_chunk",
        )
        chunks.append(chunk)

    return chunks


def chunk_text(
    content: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks of specified size.

    Args:
        content: Text content to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if not content:
        return []

    # Split into words for more natural chunking
    words = content.split()

    avg_word_length = len(content) / max(len(words), 1)
    words_per_chunk = int(chunk_size / avg_word_length)
    overlap_words = int(chunk_overlap / avg_word_length)

    words_per_chunk = max(words_per_chunk, 1)
    overlap_words = min(overlap_words, words_per_chunk - 1)

    chunks = []
    for i in range(0, len(words), words_per_chunk - overlap_words):
        if chunk_words := words[i : i + words_per_chunk]:
            chunks.append(" ".join(chunk_words))

    return chunks


def recursive_chunk_text(
    content: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into chunks using a recursive approach that tries to maintain semantic boundaries.

    Args:
        content: Text content to split
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if not content or len(content) <= chunk_size:
        return [content] if content else []

    paragraphs = re.split(r"\n\s*\n", content)

    # If we have multiple paragraphs, try to group them into chunks
    if len(paragraphs) > 1:
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap from previous chunk
                overlap_text = (
                    current_chunk[-chunk_overlap:]
                    if chunk_overlap < len(current_chunk)
                    else current_chunk
                )
                current_chunk = overlap_text + "\n\n" + para
            elif current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    return chunk_text(content, chunk_size, chunk_overlap)
