#!/usr/bin/env python
"""Document chunking module for RAG implementation

This script takes raw documents and splits them into smaller, manageable chunks
for more effective retrieval and context management with LLMs.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
data_dir = Path(__file__).parent.parent / 'data'
os.makedirs(data_dir, exist_ok=True)


def recursive_character_chunker(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into chunks of approximately chunk_size characters with overlap.
    
    This is a simple chunking strategy that splits text based on character count.
    It attempts to split at paragraph or sentence boundaries when possible.
    
    Args:
        text: The text to split into chunks
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters of overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the current chunk
        end = start + chunk_size
        
        if end >= len(text):
            # Add the last chunk and break
            chunks.append(text[start:])
            break
        
        # Try to find paragraph break
        paragraph_break = text.rfind('\n\n', start, end)
        if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
            # Split at paragraph if it's reasonably sized
            end = paragraph_break + 2  # Include the newlines
        else:
            # Try to find sentence break
            sentence_break = max(
                text.rfind('. ', start, end),
                text.rfind('! ', start, end),
                text.rfind('? ', start, end)
            )
            if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                # Split at sentence if it's reasonably sized
                end = sentence_break + 2  # Include the punctuation and space
            else:
                # Fall back to word boundary
                space = text.rfind(' ', start, end)
                if space != -1 and space > start + chunk_size // 2:
                    end = space + 1  # Include the space
        
        # Add the chunk
        chunks.append(text[start:end])
        
        # Move start for next chunk, accounting for overlap
        start = max(start + 1, end - chunk_overlap)
    
    return chunks


def semantic_chunker(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split text into semantically meaningful chunks.
    
    This chunker tries to preserve semantic sections like headings with their content.
    It's particularly effective for documentation and structured text.
    
    Args:
        text: The text to split into chunks
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters of overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split text by section headings (markdown style)
    import re
    
    # Identify headings (markdown style: # Heading)
    heading_pattern = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)
    
    # Find all headings and their positions
    headings = [(m.start(), m.group()) for m in heading_pattern.finditer(text)]
    
    # If no headings found, fall back to recursive character chunking
    if not headings:
        return recursive_character_chunker(text, chunk_size, chunk_overlap)
    
    # Add the end of text as a boundary
    headings.append((len(text), ''))
    
    # Extract sections based on headings
    sections = []
    for i in range(len(headings) - 1):
        start_pos = headings[i][0]
        end_pos = headings[i+1][0]
        section_text = text[start_pos:end_pos].strip()
        
        # Only add non-empty sections
        if section_text:
            sections.append(section_text)
    
    # Now chunk each section, preserving heading context
    chunks = []
    for section in sections:
        # If section is smaller than chunk size, add it directly
        if len(section) <= chunk_size:
            chunks.append(section)
            continue
        
        # Extract the heading
        section_lines = section.split('\n')
        heading = section_lines[0] if heading_pattern.match(section_lines[0]) else ''
        content = '\n'.join(section_lines[1:]) if heading else section
        
        # Split the content
        content_chunks = recursive_character_chunker(content, chunk_size - len(heading) - 1, chunk_overlap)
        
        # Add the heading to each chunk
        for content_chunk in content_chunks:
            if heading:
                chunks.append(f"{heading}\n{content_chunk}")
            else:
                chunks.append(content_chunk)
    
    return chunks


def chunk_documents(
    documents: List[Dict[str, Any]], 
    chunker: Optional[Callable] = None,
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    """Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of document dictionaries with 'text' field
        chunker: Chunking function to use
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters of overlap between chunks
        
    Returns:
        List of chunk dictionaries with document metadata
    """
    logger.info(f"Chunking {len(documents)} documents")
    
    # Use semantic chunker by default
    if chunker is None:
        chunker = partial(semantic_chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    chunks = []
    for doc_idx, doc in enumerate(documents):
        # Skip documents without text
        if not doc.get('text'):
            logger.warning(f"Document {doc_idx} has no text, skipping")
            continue
        
        # Get document text and metadata
        text = doc['text']
        metadata = {k: v for k, v in doc.items() if k != 'text'}
        
        # Chunk the document
        try:
            text_chunks = chunker(text)
            logger.info(f"Document {doc_idx} split into {len(text_chunks)} chunks")
            
            # Create chunk objects with metadata
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunk = {
                    'doc_id': doc.get('id', f"doc_{doc_idx}"),
                    'chunk_id': f"chunk_{doc_idx}_{chunk_idx}",
                    'text': chunk_text,
                    'chunk_idx': chunk_idx,
                    **metadata
                }
                chunks.append(chunk)
        except Exception as e:
            logger.error(f"Error chunking document {doc_idx}: {str(e)}")
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Save chunks to disk
    output_path = data_dir / 'document_chunks.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved document chunks to {output_path}")
    
    return chunks


if __name__ == "__main__":
    # Load raw documents
    raw_docs_path = data_dir / 'raw_documents.json'
    if not raw_docs_path.exists():
        logger.error(f"Raw documents not found at {raw_docs_path}")
        logger.info("Run 1_extraction.py first to extract documents")
        exit(1)
        
    with open(raw_docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    # Print summary
    print(f"\nCreated {len(chunks)} chunks from {len(documents)} documents")
    if chunks:
        print(f"Sample chunk length: {len(chunks[0].get('text', ''))} characters")
        print(f"First few words: {chunks[0].get('text', '')[:50]}...")