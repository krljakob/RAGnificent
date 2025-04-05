#!/usr/bin/env python
"""Document extraction module for RAG implementation

This script extracts content from web pages and converts them to a standardized
document format that can be used in the RAG pipeline.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Optional

from docling.document_converter import DocumentConverter
from utils.sitemap import get_sitemap_urls

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
data_dir = Path(__file__).parent.parent / 'data'
os.makedirs(data_dir, exist_ok=True)


def extract_documents(base_url: str, limit: Optional[int] = None) -> List[dict]:
    """Extract documents from a website.
    
    Args:
        base_url: The base URL to extract documents from.
        limit: Maximum number of URLs to process (None for all).
        
    Returns:
        A list of extracted documents.
    """
    logger.info(f"Extracting documents from {base_url}")
    
    # Get URLs from sitemap
    urls = get_sitemap_urls(base_url)
    logger.info(f"Found {len(urls)} URLs in sitemap")
    
    if limit:
        urls = urls[:limit]
        logger.info(f"Limited to processing {limit} URLs")
    
    # Initialize converter
    converter = DocumentConverter()
    
    # Process each URL
    successful = 0
    for i, url in enumerate(urls):
        try:
            logger.info(f"Processing [{i+1}/{len(urls)}]: {url}")
            converter.convert(url)
            successful += 1
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
    
    logger.info(f"Successfully processed {successful} out of {len(urls)} URLs")
    
    # Save to disk
    output_path = data_dir / 'raw_documents.json'
    converter.save(str(output_path))
    logger.info(f"Saved raw documents to {output_path}")
    
    # Return the documents
    with open(output_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    return documents


if __name__ == "__main__":
    # Extract documents
    base_url = "https://solana.com/docs"
    documents = extract_documents(base_url, limit=20)  # Limit for testing
    
    # Print summary
    print(f"\nExtracted {len(documents)} documents")
    if documents:
        print(f"Sample document title: {documents[0].get('title', 'No title')}")
        print(f"Sample document length: {len(documents[0].get('text', ''))} characters")
