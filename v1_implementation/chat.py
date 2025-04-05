#!/usr/bin/env python
"""Chat interface for RAG implementation

This script provides a chat interface that combines semantic search
with a language model to generate contextually informed responses.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

from dotenv import load_dotenv

# Import local modules
from search import search_chunks, format_search_results

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


def format_rag_prompt(query: str, context: str) -> str:
    """Format a prompt for RAG with retrieved context.
    
    Args:
        query: User query
        context: Retrieved context
        
    Returns:
        Formatted prompt
    """
    return f"""
You are a helpful assistant. Answer the user's question based on the provided context.
If the context doesn't contain relevant information, just say you don't know.
Don't make up information that's not in the context.

Context:
{context}

User Question: {query}

Your Answer:
"""


def get_response_from_openai(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Get a response from OpenAI's API.
    
    Args:
        prompt: Formatted prompt
        model: OpenAI model name
        
    Returns:
        Model response
    """
    try:
        return _extracted_from_get_response_from_openai_(model, prompt)
    except Exception as e:
        logger.error(f"Error getting OpenAI response: {str(e)}")
        logger.info("Falling back to local alternative")
        return get_response_from_local_model(prompt)


# TODO Rename this here and in `get_response_from_openai`
def _extracted_from_get_response_from_openai_(model, prompt):
    from openai import OpenAI

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")

    client = OpenAI(api_key=api_key)
    logger.info(f"Sending prompt to {model}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content


def chat_interface(query: str, context: Optional[str] = None) -> str:
    """Basic chat interface for RAG pipeline."""
    if context:
        prompt = format_rag_prompt(query, context)
        return get_response_from_openai(prompt)
    return get_response_from_openai(query)


def get_response_from_local_model(prompt: str) -> str:
    """Get a response from a local model or simple heuristic.
    
    This is a fallback when API models are not available.
    
    Args:
        prompt: Formatted prompt
        
    Returns:
        Model response
    """
    try:
        # Try using a local model like llama-cpp-python if installed
        from llama_cpp import Llama

        # Find models in models directory
        models_dir = Path(__file__).parent.parent / 'models'
        if model_files := list(models_dir.glob("*.gguf")):
            model_path = str(model_files[0])
            logger.info(f"Using local model: {model_path}")

            llm = Llama(model_path=model_path, n_ctx=2048)
            response = llm(prompt, max_tokens=1000, temperature=0.7, stop=["\n\n"])
            return response["choices"][0]["text"]
    except ImportError:
        logger.warning("llama-cpp-python not installed, falling back to simple summary")
    except Exception as e:
        logger.error(f"Error with local model: {str(e)}")

    # If all else fails, extract key sentences from context as a simple summary
    lines = prompt.split('\n')
    context_lines = []
    in_context = False

    for line in lines:
        if line.startswith("Context:"):
            in_context = True
            continue
        elif line.startswith("User Question:"):
            in_context = False
            question = line.replace("User Question:", "").strip()

        if in_context and line.strip():
            context_lines.append(line.strip())

    if not context_lines:
        return "I don't have enough context to answer that question."

    # Extract important sentences containing keywords from the question
    keywords = [word.lower() for word in question.split() if len(word) > 3]
    relevant_lines = [
            line
            for line in context_lines
            if any(keyword in line.lower() for keyword in keywords)
        ] or context_lines[:3]  # Just take the first few lines

    # Construct a simple response
    response = f"Based on the available information:\n\n"
    response += "\n\n".join(relevant_lines)

    return response


def chat(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
    model: str = "gpt-3.5-turbo"
) -> str:
    """Chat with RAG-enhanced responses.
    
    Args:
        query: User query
        chunks: List of chunk dictionaries with 'text' and 'embedding' fields
        top_k: Number of top results to use as context
        model: LLM model to use for responses
        
    Returns:
        Generated response
    """
    logger.info(f"Processing query: {query}")
    
    # Get relevant chunks
    search_results = search_chunks(query, chunks, top_k=top_k)
    
    if not search_results:
        return "I couldn't find any relevant information to answer your question."
    
    # Format search results as context
    context = "\n\n".join([chunk['text'] for chunk, _ in search_results])
    
    # Create prompt
    prompt = format_rag_prompt(query, context)
    
    # Get response
    if os.environ.get("OPENAI_API_KEY"):
        response = get_response_from_openai(prompt, model)
    else:
        response = get_response_from_local_model(prompt)
    
    logger.info("Generated response from model")
    
    return response


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
    
    # Interactive chat
    print("\nRAG-Powered Chat Interface")
    print("=" * 50)
    print("Ask questions about the documents (or 'quit' to exit)")
    
    model = "gpt-3.5-turbo"
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nNote: No OpenAI API key found. Using fallback response generation.")
    
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            break
            
        if not query:
            continue
        
        # Get RAG-enhanced response
        response = chat(query, chunks_with_embeddings)
        
        # Display response
        print("\nAnswer:")
        print("-" * 50)
        print(response)
        print("-" * 50)
