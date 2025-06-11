"""
Chat interface for RAGnificent.
Provides a chat interface with RAG-enhanced responses.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Use relative imports for internal modules
try:
    from .pipeline import Pipeline as RAGPipeline
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rag.pipeline import Pipeline as RAGPipeline

# Use built-in RAG functionality instead of v1_implementation
try:
    from ..core.config import get_config
except ImportError:
    from core.config import get_config

logger = logging.getLogger(__name__)


class RAGChat:
    """Chat interface with RAG-enhanced responses."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the RAG chat.

        Args:
            collection_name: Name of the vector collection
            embedding_model: Name of embedding model
        """
        self.pipeline = RAGPipeline(
            collection_name=collection_name, embedding_model=embedding_model
        )
        self.history = []

    def chat(
        self,
        query: str,
        use_history: bool = True,
        limit: int = 5,
        threshold: float = 0.7,
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
            if isinstance(result, dict):
                payload = result.get("payload", {})
                content = payload.get("content", "")
                source_url = payload.get("source_url", "")
                score = result.get("score", 0)
            else:
                # Handle SearchResult objects
                content = getattr(result, "content", "")
                source_url = getattr(result, "source_url", "")
                score = getattr(result, "score", 0)

            context.append({"content": content, "url": source_url, "score": score})

        # Combine with chat history if requested
        chat_context = ""
        if use_history and len(self.history) > 1:
            # Only include previous exchanges, not the current query
            prev_history = self.history[:-1]
            history_text = "\n".join(
                [f"{msg['role'].upper()}: {msg['content']}" for msg in prev_history]
            )
            chat_context = f"PREVIOUS CONVERSATION:\n{history_text}\n\n"

        # Generate response using built-in RAG functionality
        if context:
            response = self.pipeline.query_with_context(
                query,
                limit=limit,
                threshold=threshold,
                system_prompt=chat_context or None,
            )
            # Extract the response text if it's a dict
            if isinstance(response, dict):
                response = response.get("response", str(response))
        else:
            # No context found
            response = "I couldn't find relevant information to answer your query. Could you rephrase or ask something else?"

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})

        return {
            "query": query,
            "response": response,
            "context": context,
            "has_context": bool(context),
        }

    def clear_history(self):
        """Clear chat history."""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get chat history."""
        return self.history
