"""
Chat interface for RAGnificent.
Provides a chat interface with RAG-enhanced responses.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from RAGnificent.rag.pipeline import RAGPipeline

# Use v1_implementation's agent functionality
from v1_implementation.agent import query_with_context, summarize_documents
from v1_implementation.chat import format_message

logger = logging.getLogger(__name__)

class RAGChat:
    """Chat interface with RAG-enhanced responses."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the RAG chat.

        Args:
            collection_name: Name of the vector collection
            embedding_model: Name of embedding model
        """
        self.pipeline = RAGPipeline(
            collection_name=collection_name,
            embedding_model=embedding_model
        )
        self.history = []
        
    def chat(
        self, 
        query: str,
        use_history: bool = True,
        limit: int = 5,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Chat with RAG-enhanced responses.

        Args:
            query: User query
            use_history: Whether to use chat history
            limit: Maximum number of results
            threshold: Similarity threshold

        Returns:
            Response dictionary with query, response, and context
        """
        logger.info(f"RAG chat query: {query}")
        
        # Add user message to history
        self.history.append({"role": "user", "content": query})
        
        # Get relevant context for query
        results = self.pipeline.search_documents(query, limit, threshold)
        
        # Format context for the agent
        context = []
        for result in results:
            payload = result.get("payload", {})
            content = payload.get("content", "")
            source_url = payload.get("source_url", "")
            
            context.append({
                "content": content,
                "url": source_url,
                "score": result.get("score", 0)
            })
            
        # Combine with chat history if requested
        chat_context = ""
        if use_history and len(self.history) > 1:
            # Only include previous exchanges, not the current query
            prev_history = self.history[:-1]
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in prev_history
            ])
            chat_context = f"PREVIOUS CONVERSATION:\n{history_text}\n\n"
            
        # Generate response using agent
        if context:
            response = query_with_context(
                query, 
                context,
                system_prompt=chat_context if chat_context else None
            )
        else:
            # No context found
            response = "I couldn't find relevant information to answer your query. Could you rephrase or ask something else?"
            
        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})
        
        return {
            "query": query,
            "response": response,
            "context": context,
            "has_context": bool(context)
        }
        
    def clear_history(self):
        """Clear chat history."""
        self.history = []
        
    def get_history(self) -> List[Dict[str, str]]:
        """Get chat history."""
        return self.history
