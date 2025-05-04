#!/usr/bin/env python
"""Document chunking module for RAG implementation

This script takes raw documents and splits them into smaller, manageable chunks
for more effective retrieval and context management with LLMs. It includes robust
error handling, input validation, and proper logging for reliable operation.
"""
import hashlib
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, validator

from config import ChunkingConfig, ChunkingStrategy, load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
data_dir = Path(__file__).parent.parent / 'data'
os.makedirs(data_dir, exist_ok=True)

# Data models for validation
class ChunkingOptions(BaseModel):
    """Options for document chunking"""
    strategy: ChunkingStrategy = Field(
        ChunkingStrategy.SEMANTIC,
        description="Chunking strategy to use"
    )
    chunk_size: int = Field(1000, ge=100, le=10000, description="Target chunk size in characters")
    chunk_overlap: int = Field(200, ge=0, description="Overlap between chunks in characters")
    separator: str = Field("\n", description="Default separator for boundary decisions")
    keep_separator: bool = Field(True, description="Whether to keep separators in chunks")
    min_chunk_size: int = Field(100, ge=50, description="Minimum allowed chunk size")
    max_chunk_size: int = Field(2000, le=20000, description="Maximum allowed chunk size")
    
    @validator('chunk_overlap')
    def overlap_less_than_size(cls, v, values):
        """Validate chunk overlap is less than chunk size"""
        if 'chunk_size' in values.data and v >= values.data['chunk_size']:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({values.data['chunk_size']})")
        return v
    
    @validator('min_chunk_size')
    def min_less_than_max(cls, v, values):
        """Validate min chunk size is less than max"""
        if 'max_chunk_size' in values.data and v >= values.data['max_chunk_size']:
            raise ValueError(f"Min chunk size ({v}) must be less than max chunk size ({values.data['max_chunk_size']})")
        return v


class ChunkingError(Exception):
    """Base exception for chunking errors"""
    pass


class ChunkingInputError(ChunkingError):
    """Exception for invalid chunking inputs"""
    pass


class ChunkingSizeError(ChunkingError):
    """Exception for chunking size problems"""
    pass


class Document(BaseModel):
    """Document model for validation"""
    id: Optional[str] = Field(None, description="Document ID")
    text: str = Field(..., description="Document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    
    @validator('text')
    def text_not_empty(cls, v):
        """Validate text is not empty"""
        if not v or not v.strip():
            raise ValueError("Document text cannot be empty")
        return v
    
    def generate_id(self) -> str:
        """Generate a stable ID based on content hash if not provided"""
        if not self.id:
            # Create a hash from the first 100 chars of text and metadata
            content = self.text[:100] + str(sorted(self.metadata.items()))
            self.id = hashlib.md5(content.encode()).hexdigest()[:16]
        return self.id


def recursive_character_chunker(text: str, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = "\n") -> List[str]:
    """Split text into chunks of approximately chunk_size characters with overlap.

    This is a simple chunking strategy that splits text based on character count.
    It attempts to split at paragraph or sentence boundaries when possible.

    Args:
        text: The text to split into chunks
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters of overlap between chunks
        separator: Preferred separator for chunking boundaries

    Returns:
        List of text chunks
        
    Raises:
        ValueError: If chunking parameters are invalid
        ChunkingSizeError: If chunk size is too small for the text
    """
    # Validate parameters
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size")
    if not text:
        return []
        
    # If text is smaller than chunk size, return as single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    
    # Track some metrics for logging
    total_chars = len(text)
    processed_chars = 0
    
    logger.debug(f"Chunking text of length {total_chars} with chunk_size={chunk_size}, overlap={chunk_overlap}")

    try:
        while start < len(text):
            # Find the end of the current chunk
            end = min(start + chunk_size, len(text))
            processed_chars = max(processed_chars, end)
            
            if end >= len(text):
                # Add the last chunk and break
                chunks.append(text[start:])
                break

            # Find the best boundary for this chunk
            # Try to find paragraph break (two consecutive newlines)
            paragraph_break = text.rfind('\n\n', start, end)
            
            # If separator is specified and different from newline, also try it
            if separator != '\n' and separator in text:
                sep_break = text.rfind(separator, start, end)
            else:
                sep_break = -1
                
            # Use the best boundary found (paragraph, separator, or none)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                # Split at paragraph if it's reasonably sized
                end = paragraph_break + 2  # Include the newlines
                logger.debug(f"Chunk boundary: paragraph at position {end}")
            elif sep_break != -1 and sep_break > start + chunk_size // 2:
                # Split at separator if it's reasonably sized
                end = sep_break + len(separator)  # Include the separator
                logger.debug(f"Chunk boundary: separator at position {end}")
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
                    logger.debug(f"Chunk boundary: sentence at position {end}")
                else:
                    # Fall back to word boundary
                    space = text.rfind(' ', start, end)
                    if space != -1 and space > start + chunk_size // 2:
                        end = space + 1  # Include the space
                        logger.debug(f"Chunk boundary: word at position {end}")
                    else:
                        # If all else fails, just split at the chunk size
                        logger.debug(f"Chunk boundary: character at position {end}")

            # Add the chunk
            chunk = text[start:end]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start for next chunk, accounting for overlap
            start = max(start + 1, end - chunk_overlap)
            
        # Log chunking statistics
        avg_chunk_size = sum(len(c) for c in chunks) / max(1, len(chunks))
        logger.debug(f"Created {len(chunks)} chunks with average size {avg_chunk_size:.1f} characters")
        return chunks
        
    except Exception as e:
        logger.error(f"Error during recursive chunking: {str(e)}")
        raise ChunkingError(f"Failed to chunk text: {str(e)}") from e


def semantic_chunker(text: str, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = "\n") -> List[str]:
    """Split text into semantically meaningful chunks based on document structure.

    This chunker tries to preserve semantic sections like headings with their content.
    It's particularly effective for documentation and structured text. It uses an
    intelligent approach to identify section boundaries and preserves the document
    hierarchy in chunks.

    Args:
        text: The text to split into chunks
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters of overlap between chunks
        separator: Custom separator for boundaries (default: newline)

    Returns:
        List of text chunks
        
    Raises:
        ChunkingError: If chunking fails
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if not text or not text.strip():
        logger.warning("Empty text provided to semantic chunker")
        return []
        
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
        
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be less than chunk size")
        
    # Avoid chunking very small texts
    if len(text) <= chunk_size:
        return [text]
    
    try:
        import re

        # Track statistics
        start_time = datetime.now()
        logger.debug(f"Starting semantic chunking of {len(text)} characters")

        # Identify headings (supports markdown style # Heading and HTML <h1>-<h6>)
        heading_patterns = [
            # Markdown headings (e.g., # Heading)
            re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
            # HTML headings (e.g., <h1>Heading</h1>)
            re.compile(r'<h[1-6][^>]*>.*?</h[1-6]>', re.DOTALL | re.IGNORECASE),
            # Other potential section markers like numbered sections (e.g., "1.2.3 Section")
            re.compile(r'^\d+(\.\d+)*\s+[A-Z].*$', re.MULTILINE)
        ]

        # Find all potential section boundaries
        all_headers = []
        for pattern in heading_patterns:
            matches = list(pattern.finditer(text))
            all_headers.extend([(m.start(), m.group()) for m in matches])

        # Sort headers by position
        all_headers.sort(key=lambda x: x[0])
        
        # Debug info on headers found
        if all_headers:
            logger.debug(f"Found {len(all_headers)} potential section headers")
        else:
            logger.debug("No section headers found, falling back to character chunking")
            return recursive_character_chunker(text, chunk_size, chunk_overlap, separator)

        # Add the end of text as a boundary
        all_headers.append((len(text), ''))

        # Extract sections based on headings
        sections = []
        for i in range(len(all_headers) - 1):
            start_pos = all_headers[i][0]
            end_pos = all_headers[i+1][0]
            
            # Skip empty sections or sections that are too close together
            if end_pos - start_pos < 20:  # Minimum section size
                continue
                
            if section_text := text[start_pos:end_pos].strip():
                sections.append(section_text)

        # If no valid sections found, fall back to character chunking
        if not sections:
            logger.debug("No valid sections extracted, falling back to character chunking")
            return recursive_character_chunker(text, chunk_size, chunk_overlap, separator)

        # Now chunk each section, preserving heading context
        chunks = []
        for i, section in enumerate(sections):
            logger.debug(f"Processing section {i+1}/{len(sections)} - size: {len(section)} characters")
            
            # If section is smaller than chunk size, add it directly
            if len(section) <= chunk_size:
                chunks.append(section)
                continue

            # Extract the heading - more flexible approach
            # Try to identify the heading from the section
            section_lines = section.split('\n')
            heading = ""
            
            # Check if first line matches any heading pattern
            first_line = section_lines[0] if section_lines else ""
            for pattern in heading_patterns:
                if pattern.match(first_line):
                    heading = first_line
                    content = '\n'.join(section_lines[1:]) if len(section_lines) > 1 else ""
                    break
            else:
                # No heading pattern matched, use the whole section
                content = section

            # Calculate effective chunk size (reduced if we have a heading)
            effective_chunk_size = chunk_size
            if heading:
                # Reserve space for the heading in each chunk
                heading_len = len(heading) + 1  # +1 for newline
                effective_chunk_size = max(100, chunk_size - heading_len)

            # Split the content
            try:
                content_chunks = recursive_character_chunker(
                    content, 
                    chunk_size=effective_chunk_size, 
                    chunk_overlap=chunk_overlap,
                    separator=separator
                )
                
                # Add the heading to each chunk
                for j, content_chunk in enumerate(content_chunks):
                    if heading:
                        # Include section number in metadata for reassembly if needed
                        chunk_with_heading = f"{heading}\n{content_chunk}"
                        chunks.append(chunk_with_heading)
                    else:
                        chunks.append(content_chunk)
                        
            except Exception as e:
                logger.warning(f"Error chunking section {i+1}: {str(e)}, skipping to next section")
                # Still include the section as a single chunk if possible
                if len(section) <= chunk_size * 2:  # Only include if reasonably sized
                    chunks.append(section)

        # Calculate statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        if chunks:
            avg_chunk_size = sum(len(c) for c in chunks) / len(chunks)
            logger.debug(f"Created {len(chunks)} semantic chunks with average size {avg_chunk_size:.1f} chars in {elapsed:.2f}s")
        else:
            logger.warning("No chunks were created during semantic chunking")

        return chunks
        
    except Exception as e:
        logger.error(f"Error during semantic chunking: {str(e)}")
        logger.error(traceback.format_exc())
        # Fall back to character chunking as last resort
        logger.warning("Falling back to character chunking due to error")
        try:
            return recursive_character_chunker(text, chunk_size, chunk_overlap, separator)
        except Exception as fallback_error:
            logger.error(f"Even fallback chunking failed: {str(fallback_error)}")
            raise ChunkingError(f"Failed to chunk text: {str(e)}")


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunker: Optional[Callable] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    separator: str = "\n",
    config: Optional[ChunkingConfig] = None,
    output_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """Split documents into smaller chunks for better retrieval and context management.

    Args:
        documents: List of document dictionaries with 'text' field
        chunker: Custom chunking function to use
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters of overlap between chunks
        strategy: Chunking strategy to use
        separator: Custom separator for boundary decisions
        config: Optional chunking configuration
        output_path: Optional path to save chunked documents

    Returns:
        List of chunk dictionaries with document metadata
        
    Raises:
        ValueError: If input parameters are invalid
        ChunkingError: If chunking process fails
        IOError: If saving chunks fails
    """
    start_time = datetime.now()
    
    # Validate and normalize input documents
    validated_docs = []
    invalid_docs = 0
    empty_docs = 0
    
    # Load config if not provided
    if config is None:
        try:
            app_config = load_config()
            config = app_config.chunking
            logger.debug("Using default chunking configuration")
        except Exception as e:
            logger.warning(f"Failed to load config: {str(e)}. Using default values.")
            config = ChunkingConfig()

    # Override parameters with config if provided
    if config:
        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap
        strategy = config.strategy
        separator = config.separator
    
    # Validate documents
    for doc_idx, doc in enumerate(documents):
        try:
            # Skip empty documents
            if not doc.get('text'):
                logger.warning(f"Document {doc_idx} has no text content, skipping")
                empty_docs += 1
                continue
                
            # Convert to Document model for validation
            validated_doc = Document(
                id=doc.get('id'),
                text=doc.get('text', ""),
                metadata={k: v for k, v in doc.items() if k not in ['id', 'text']}
            )
            
            # Generate ID if not present
            if not validated_doc.id:
                validated_doc.generate_id()
                
            validated_docs.append(validated_doc)
            
        except ValidationError as e:
            logger.warning(f"Document {doc_idx} validation failed: {str(e)}")
            invalid_docs += 1
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing document {doc_idx}: {str(e)}")
            invalid_docs += 1
            continue
    
    if not validated_docs:
        raise ChunkingInputError(f"No valid documents to chunk. Invalid: {invalid_docs}, Empty: {empty_docs}")
        
    logger.info(f"Chunking {len(validated_docs)} valid documents (skipped: {invalid_docs + empty_docs})")
    
    # Choose chunking function based on strategy
    if chunker is None:
        if strategy == ChunkingStrategy.SEMANTIC:
            chunker_func = partial(
                semantic_chunker, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separator=separator
            )
            logger.info(f"Using semantic chunking strategy (size={chunk_size}, overlap={chunk_overlap})")
            
        elif strategy == ChunkingStrategy.RECURSIVE:
            chunker_func = partial(
                recursive_character_chunker, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separator=separator
            )
            logger.info(f"Using recursive character chunking strategy (size={chunk_size}, overlap={chunk_overlap})")
            
        elif strategy == ChunkingStrategy.SLIDING_WINDOW:
            # Implement a simple sliding window chunker
            def sliding_window_chunker(text, window_size=chunk_size, stride=chunk_size - chunk_overlap):
                chunks = []
                for i in range(0, len(text), stride):
                    if i + window_size >= len(text):
                        # Last chunk may be smaller
                        chunks.append(text[i:])
                        break
                    chunks.append(text[i:i + window_size])
                return chunks
                
            chunker_func = sliding_window_chunker
            logger.info(f"Using sliding window chunking strategy (size={chunk_size}, stride={chunk_size - chunk_overlap})")
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    else:
        # Use provided custom chunker
        chunker_func = chunker
        logger.info("Using custom chunking function")

    # Process each document
    all_chunks = []
    total_chunks = 0
    successful_docs = 0
    failed_docs = 0
    
    # Track chunk distribution for diagnostics
    chunk_sizes = []
    chunks_per_doc = []

    try:
        for doc_idx, doc in enumerate(validated_docs):
            try:
                # Log progress for large document sets
                if (doc_idx + 1) % 10 == 0 or doc_idx == 0:
                    logger.info(f"Processing document {doc_idx + 1}/{len(validated_docs)}")
                else:
                    logger.debug(f"Processing document {doc_idx + 1}/{len(validated_docs)}")
                
                # Get document text and metadata
                # Convert model to dict for processing
                doc_dict = doc.model_dump()
                text = doc_dict['text']
                metadata = doc_dict.get('metadata', {})
                doc_id = doc_dict['id']
                
                # Apply chunking with proper error handling
                try:
                    text_chunks = chunker_func(text)
                    if not text_chunks:
                        logger.warning(f"Document {doc_id} resulted in no chunks, skipping")
                        continue
                        
                    # Log success
                    logger.debug(f"Document {doc_id} split into {len(text_chunks)} chunks")
                    chunks_per_doc.append(len(text_chunks))
                    
                    # Create chunk objects with metadata
                    doc_chunks = []
                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        # Skip empty chunks
                        if not chunk_text.strip():
                            continue
                            
                        # Track chunk sizes for diagnostics
                        chunk_sizes.append(len(chunk_text))
                        
                        # Create the chunk with document metadata
                        chunk = {
                            'doc_id': doc_id,
                            'chunk_id': f"{doc_id}_chunk_{chunk_idx}",
                            'text': chunk_text,
                            'chunk_idx': chunk_idx,
                            'chunk_count': len(text_chunks),  # Total chunks in the document
                            'metadata': metadata,
                            'created_at': datetime.now().isoformat()
                        }
                        doc_chunks.append(chunk)
                        
                    # Add all chunks from this document
                    all_chunks.extend(doc_chunks)
                    total_chunks += len(doc_chunks)
                    successful_docs += 1
                    
                except Exception as chunk_error:
                    logger.error(f"Error chunking document {doc_id}: {str(chunk_error)}")
                    failed_docs += 1
                    # Continue processing other documents
                    
            except Exception as doc_error:
                logger.error(f"Error processing document at index {doc_idx}: {str(doc_error)}")
                failed_docs += 1
                continue
    
        # Calculate statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if chunk_sizes:
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
            max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
        else:
            avg_chunk_size = min_chunk_size = max_chunk_size = 0
            
        if chunks_per_doc:
            avg_chunks_per_doc = sum(chunks_per_doc) / len(chunks_per_doc)
        else:
            avg_chunks_per_doc = 0
            
        logger.info(f"Chunking completed in {elapsed:.2f}s")
        logger.info(f"Created {total_chunks} chunks from {successful_docs} documents (failed: {failed_docs})")
        logger.info(f"Chunk size: avg={avg_chunk_size:.1f}, min={min_chunk_size}, max={max_chunk_size}")
        logger.info(f"Chunks per document: avg={avg_chunks_per_doc:.1f}")
        
        # Save chunks to file if output path specified
        if output_path or (output_path is None and len(all_chunks) > 0):
            # Determine output path
            if output_path:
                chunk_path = Path(output_path) if isinstance(output_path, str) else output_path
            else:
                # Create default path with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                chunk_path = data_dir / f'document_chunks_{timestamp}.json'
                
            # Ensure directory exists
            os.makedirs(chunk_path.parent, exist_ok=True)
            
            # Save with error handling
            try:
                with open(chunk_path, 'w', encoding='utf-8') as f:
                    # Add metadata for the chunks
                    output_data = {
                        'chunks': all_chunks,
                        'metadata': {
                            'document_count': len(validated_docs),
                            'chunk_count': total_chunks,
                            'chunk_size': chunk_size,
                            'chunk_overlap': chunk_overlap,
                            'strategy': strategy.value,
                            'created_at': datetime.now().isoformat()
                        }
                    }
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {total_chunks} chunks to {chunk_path}")
            except Exception as save_error:
                logger.error(f"Failed to save chunks to {chunk_path}: {str(save_error)}")
                raise IOError(f"Failed to save chunks: {str(save_error)}") from save_error
        
        return all_chunks
        
    except Exception as e:
        logger.error(f"Chunking process failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise ChunkingError(f"Document chunking failed: {str(e)}") from e

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
        sys.exit(1)

    with open(raw_docs_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Chunk documents
    chunks = chunk_documents(documents)
