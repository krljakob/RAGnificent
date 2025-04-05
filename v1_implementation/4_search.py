#!/usr/bin/env python
"""Semantic search module for RAG implementation

This script provides search functionality to find relevant document chunks
based on semantic similarity to a query.
"""
import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np
from dotenv import load_dotenv

# Import local modules
from embedding import get_openai_embeddings, get_local_embeddings

# Load environment variables for API keys
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure data directory exists
data_dir = Path(__file__).parent.parent / 'data'
os.makedirs(data_dir, exist_ok=True)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (float between -1 and 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def batch_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between query and multiple document vectors.
    
    Args:
        query_vec: Query vector [embedding_dim]
        doc_vecs: Document vectors [num_docs, embedding_dim]
        
    Returns:
        Array of similarity scores [num_docs]
    """
    # Normalize vectors
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    
    # Avoid division by zero
    query_norm = max(query_norm, 1e-10)
    doc_norms = np.maximum(doc_norms, 1e-10)
    
    # Calculate dot product
    dot_products = np.dot(doc_vecs, query_vec)
    
    # Calculate similarities
    similarities = dot_products / (doc_norms * query_norm)
    
    return similarities


def search_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    embedding_function: Optional[Callable] = None,
    top_k: int = 5
) -> List[Tuple[Dict[str, Any], float]]:
    """Search for chunks similar to query.
    
    Args:
        query: Search query
        chunks: List of chunk dictionaries with 'text' and 'embedding' fields
        embedding_function: Function to generate query embedding
        top_k: Number of top results to return
        
    Returns:
        List of (chunk, similarity_score) tuples
    """
    logger.info(f"Searching for query: {query}")
    
    # Filter chunks to only those with embeddings
    chunks_with_embeddings = [chunk for chunk in chunks if 'embedding' in chunk]
    
    if not chunks_with_embeddings:
        logger.warning("No chunks with embeddings found")
        return []
    
    # Use appropriate embedding function
    if embedding_function is None:
        # Try OpenAI first, fall back to local methods
        if os.environ.get("OPENAI_API_KEY"):
            embedding_function = get_openai_embeddings
        else:
            embedding_function = get_local_embeddings
    
    # Generate query embedding
    try:
        query_embedding = embedding_function([query])[0]
    except Exception as e:
        logger.error(f"Error generating query embedding: {str(e)}")
        
        # Fall back to TF-IDF if we have the vectorizer saved
        try:
            vectorizer_path = data_dir / 'tfidf_vectorizer.pkl'
            if vectorizer_path.exists():
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                query_embedding = vectorizer.transform([query]).toarray()[0]
            else:
                logger.error("No TF-IDF vectorizer found for fallback")
                return []
        except Exception as e2:
            logger.error(f"Error with TF-IDF fallback: {str(e2)}")
            return []
    
    # Get document embeddings
    doc_embeddings = np.array([chunk['embedding'] for chunk in chunks_with_embeddings])
    
    # Calculate similarities
    similarities = batch_cosine_similarity(query_embedding, doc_embeddings)
    
    # Sort by similarity
    results = []
    for i, similarity in enumerate(similarities):
        results.append((chunks_with_embeddings[i], float(similarity)))
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k results
    top_results = results[:top_k]
    logger.info(f"Found {len(top_results)} relevant chunks")
    
    return top_results


def format_search_results(results: List[Tuple[Dict[str, Any], float]]) -> str:
    """Format search results for display.
    
    Args:
        results: List of (chunk, similarity_score) tuples
        
    Returns:
        Formatted string of results
    """
    if not results:
        return "No relevant documents found."
    
    formatted = "Search Results:\n" + "=" * 50 + "\n\n"
    
    for i, (chunk, score) in enumerate(results, 1):
        # Get chunk information
        title = chunk.get('title', 'Untitled')
        url = chunk.get('url', '')
        text = chunk.get('text', '')
        
        # Format chunk
        formatted += f"Result {i} (Relevance: {score:.4f})\n"
        formatted += f"Title: {title}\n"
        if url:
            formatted += f"Source: {url}\n"
        formatted += "\n"
        
        # Show preview of text (first 300 chars)
        preview = text[:300] + "..." if len(text) > 300 else text
        formatted += f"{preview}\n\n"
        formatted += "-" * 50 + "\n\n"
    
    return formatted


if __name__ == "__main__":
    # Load embedded chunks
    chunks_path = data_dir / 'embedded_chunks.json'
    if not chunks_path.exists():
        logger.error(f"Embedded chunks not found at {chunks_path}")
        logger.info("Run 3_embedding.py first to create embeddings")
        exit(1)
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Verify chunks have embeddings
    chunks_with_embeddings = [chunk for chunk in chunks if 'embedding' in chunk]
    logger.info(f"Loaded {len(chunks_with_embeddings)} chunks with embeddings")
    
    if not chunks_with_embeddings:
        logger.error("No chunks have embeddings")
        exit(1)
    
    # Interactive search
    print("\nRAG Semantic Search Interface")
    print("=" * 50)
    print("Enter your search query (or 'quit' to exit)")
    
    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        if not query:
            continue
        
        # Search for chunks
        results = search_chunks(query, chunks)
        
        # Display results
        print("\n" + format_search_results(results))