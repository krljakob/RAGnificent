#!/usr/bin/env python
"""Document embedding module for RAG implementation

This script converts text chunks into vector embeddings for semantic search.
Multiple embedding models are supported, with a fallback to simpler models
when API keys are not available.
"""
import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable

import numpy as np
from dotenv import load_dotenv

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


def get_openai_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Generate embeddings using OpenAI's API.
    
    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model to use
        
    Returns:
        Numpy array of embeddings, shape [num_texts, embedding_dim]
    """
    try:
        from openai import OpenAI
        
        # Get API key
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        client = OpenAI(api_key=api_key)
        logger.info(f"Getting embeddings for {len(texts)} texts using OpenAI model {model}")
        
        # Process in batches of 100 to avoid rate limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            
            # Extract embeddings from response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        # Convert to numpy array
        embeddings = np.array(all_embeddings)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        return embeddings
        
    except (ImportError, ValueError, Exception) as e:
        logger.error(f"Error generating OpenAI embeddings: {str(e)}")
        logger.info("Falling back to local embedding model")
        return get_local_embeddings(texts)


def get_sentence_transformer_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Generate embeddings using Sentence Transformers locally.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the sentence-transformers model
        
    Returns:
        Numpy array of embeddings, shape [num_texts, embedding_dim]
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading Sentence Transformers model: {model_name}")
        model = SentenceTransformer(model_name)
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        return embeddings
        
    except ImportError:
        logger.error("Sentence Transformers not installed")
        logger.info("Installing now...")
        os.system("pip install sentence-transformers")
        
        # Retry after installation
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error installing or using Sentence Transformers: {str(e)}")
            logger.info("Falling back to TF-IDF embeddings")
            return get_tfidf_embeddings(texts)


def get_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """Generate TF-IDF embeddings as a fallback method.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        Numpy array of embeddings, shape [num_texts, vocab_size]
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    logger.info(f"Generating TF-IDF embeddings for {len(texts)} texts")
    
    # Create vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform
    embeddings = vectorizer.fit_transform(texts).toarray()
    
    logger.info(f"Generated TF-IDF embeddings with shape {embeddings.shape}")
    
    # Also save the vectorizer for later use in search
    vectorizer_path = data_dir / 'tfidf_vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved TF-IDF vectorizer to {vectorizer_path}")
    
    return embeddings


def get_local_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings using locally available models.
    
    This is a convenience function that attempts to use the best
    locally available embedding method.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        Numpy array of embeddings
    """
    # Try sentence-transformers first
    try:
        return get_sentence_transformer_embeddings(texts)
    except Exception as e:
        logger.error(f"Error with Sentence Transformers: {str(e)}")
        return get_tfidf_embeddings(texts)


def embed_chunks(
    chunks: List[Dict[str, Any]],
    embedding_function: Optional[Callable] = None,
    chunk_key: str = 'text'
) -> List[Dict[str, Any]]:
    """Embed text chunks using the specified embedding function.
    
    Args:
        chunks: List of chunk dictionaries with 'text' field
        embedding_function: Function to generate embeddings
        chunk_key: Key in chunk dictionaries containing the text to embed
        
    Returns:
        List of chunk dictionaries with added 'embedding' field
    """
    logger.info(f"Embedding {len(chunks)} chunks")
    
    # Extract texts to embed
    texts = [chunk[chunk_key] for chunk in chunks if chunk_key in chunk]
    
    # Skip if no texts to embed
    if not texts:
        logger.warning(f"No texts found to embed with key '{chunk_key}'")
        return chunks
    
    # Use appropriate embedding function
    if embedding_function is None:
        # Try OpenAI first, fall back to local methods
        if os.environ.get("OPENAI_API_KEY"):
            embedding_function = get_openai_embeddings
        else:
            embedding_function = get_local_embeddings
    
    # Generate embeddings
    try:
        embeddings = embedding_function(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            if chunk_key in chunk:
                chunk['embedding'] = embeddings[i].tolist()
        
        logger.info(f"Successfully embedded {len(texts)} chunks")
    except Exception as e:
        logger.error(f"Error embedding chunks: {str(e)}")
    
    # Save embedded chunks
    output_path = data_dir / 'embedded_chunks.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved embedded chunks to {output_path}")
    
    return chunks


if __name__ == "__main__":
    # Load chunked documents
    chunks_path = data_dir / 'document_chunks.json'
    if not chunks_path.exists():
        logger.error(f"Document chunks not found at {chunks_path}")
        logger.info("Run 2_chunking.py first to create document chunks")
        exit(1)
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Embed chunks
    embedded_chunks = embed_chunks(chunks)
    
    # Print summary
    num_with_embeddings = sum(1 for chunk in embedded_chunks if 'embedding' in chunk)
    print(f"\nEmbedded {num_with_embeddings} out of {len(embedded_chunks)} chunks")
    
    if num_with_embeddings > 0:
        sample_chunk = next(chunk for chunk in embedded_chunks if 'embedding' in chunk)
        embedding_dim = len(sample_chunk['embedding'])
        print(f"Embedding dimension: {embedding_dim}")
        
        # Print first few dimensions of first embedding
        print(f"Sample embedding values: {sample_chunk['embedding'][:5]}...")