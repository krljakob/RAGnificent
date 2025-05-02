import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize data directory
DATA_DIR = Path(__file__).parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

class RAGAgent:
    """Agent for running RAG pipeline components."""

    def __init__(self):
        self.tools = self.get_tools()

    def get_tools(self) -> List[callable]:
        """Register all available pipeline tools."""
        return [
            self.run_extraction,
            self.run_chunking,
            self.run_embedding,
            self.run_search,
            self.run_chat
        ]

    def run_extraction(self, url: str, limit: int = 20) -> str:
        """Extract documents from a URL.

        Args:
            url: Website URL to scrape
            limit: Max documents to extract

        Returns:
            Path to documents file
        """
        from extraction import extract_documents
        try:
            # Extract documents from the URL
            result = extract_documents(url, limit)

            # The function might return a list of documents or a file path
            if isinstance(result, list):
                # If it's a list, save it to a file
                output_path = str(DATA_DIR / 'raw_documents.json')
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Extracted documents saved to {output_path}")
                return output_path
            # If it's already a file path, return it
            logger.info(f"Extracted documents saved to {result}")
            return str(result)
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            # Try to use existing documents if available
            default_docs = str(DATA_DIR / 'raw_documents.json')
            if Path(default_docs).exists():
                logger.info(f"Using existing documents from {default_docs}")
                return default_docs
            raise

    def run_chunking(self, documents_path: str) -> str:
        """Chunk extracted documents.

        Args:
            documents_path: Path to documents JSON file

        Returns:
            Path to chunks JSON file
        """
        from chunking import chunk_documents
        try:
            return self._extracted_from_run_chunking_13(documents_path, chunk_documents)
        except Exception as e:
            logger.error(f"Chunking failed: {str(e)}")
            raise

    # TODO Rename this here and in `run_chunking`
    def _extracted_from_run_chunking_13(self, documents_path, chunk_documents):
        # Load documents from file
        with open(documents_path) as f:
            documents = json.load(f)

        # Chunk documents
        chunks = chunk_documents(documents)

        # Save chunks to file
        output_path = Path(documents_path).parent / 'document_chunks.json'
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)

        logger.info(f"Chunks saved to {output_path}")
        return str(output_path)

    def run_embedding(self, chunks_path: str) -> str:
        """Generate embeddings for document chunks.

        Args:
            chunks_path: Path to chunks JSON file

        Returns:
            Path to embeddings file
        """
        try:
            # Validate input
            chunks_path = Path(chunks_path)
            if not chunks_path.exists():
                raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

            # Load chunks
            with open(chunks_path) as f:
                chunks = json.load(f)

            if not isinstance(chunks, list):
                raise ValueError("Chunks data should be a list")

            texts = [chunk.get('text', '') for chunk in chunks]
            if not all(texts):
                raise ValueError("Some chunks are missing text content")

            # Create simple TF-IDF vectors as embeddings
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=100)
                tfidf_matrix = vectorizer.fit_transform(texts)
                # Convert sparse matrix to regular list of lists
                embeddings = tfidf_matrix.toarray().tolist()
                logger.info("Generated TF-IDF embeddings")
            except ImportError:
                # Fallback to simple word count vectors
                logger.warning("scikit-learn not available, using simple word count embeddings")
                embeddings = []
                for text in texts:
                    # Create a simple vector of word counts (limited to 100 dimensions)
                    words = text.lower().split()
                    word_counts = {}
                    for word in words:
                        word_counts[word] = word_counts.get(word, 0) + 1
                    # Take top 100 words by count
                    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:100]
                    # Create a simple embedding vector
                    vector = [count for _, count in top_words]
                    # Pad to exactly 100 dimensions
                    vector += [0] * (100 - len(vector))
                    embeddings.append(vector)
                logger.info("Generated simple word count embeddings")

            # Save embeddings
            output_path = chunks_path.parent / 'embeddings.json'
            with open(output_path, 'w') as f:
                json.dump({
                    'chunks': chunks,
                    'embeddings': embeddings,
                    'version': '1.0',
                    'created_at': datetime.datetime.now().isoformat()
                }, f, indent=2)

            logger.info(f"Embeddings saved to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Embedding failed: {str(e)}")
            raise

    def run_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for chunks relevant to a query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of relevant chunks
        """
        from search import search_chunks
        try:
            results = search_chunks(query, top_k=top_k)

            # Format the results for better display
            formatted_results = []
            for chunk, score in results:
                # Extract the most relevant fields
                formatted_chunk = {
                    'text': chunk.get('text', ''),
                    'url': chunk.get('url', ''),
                    'title': chunk.get('title', ''),
                    'score': f"{score:.2f}"
                }
                formatted_results.append(formatted_chunk)

            logger.info(f"Found {len(formatted_results)} relevant chunks")
            return formatted_results
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []  # Return empty list instead of raising to keep the chat flow going

    def run_chat(self, query: str, context: Optional[str] = None) -> str:
        """Get a chat response based on a query and optional context.

        Args:
            query: User query
            context: Optional context for RAG

        Returns:
            Chat response
        """
        from chat import chat_interface
        try:
            # If we have search results as context, format them nicely
            if isinstance(context, list):
                formatted_context = "\n\n".join([
                    f"Title: {chunk.get('title')}\n" +
                    f"URL: {chunk.get('url')}\n" +
                    f"Relevance: {chunk.get('score')}\n" +
                    f"Content: {chunk.get('text')}\n"
                    for chunk in context
                ])
                context = formatted_context if formatted_context.strip() else None

            return chat_interface(query, context=context)
        except Exception as e:
            logger.error(f"Chat failed: {str(e)}")
            return f"I'm sorry, I couldn't generate a response: {str(e)}"

# Example usage
def check_dependencies():
    """Check required dependencies for the RAG pipeline."""
    dependencies = {
        'bs4': "beautifulsoup4",  # For extraction
        'sklearn': "scikit-learn"  # For embeddings
    }

    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            logger.info(f"{module} is available")
        except ImportError:
            missing.append((module, package))

    return not missing

if __name__ == "__main__":
    if not check_dependencies():
        import sys
        sys.exit(1)

    # Create the agent
    agent = RAGAgent()

    # Default document path
    default_chunks_path = str(DATA_DIR / 'document_chunks.json')
    default_embeddings_path = str(DATA_DIR / 'embeddings.json')

    # Check if user wants to extract new documents or use existing
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-extraction":
        if Path(default_embeddings_path).exists():
            embeddings_path = default_embeddings_path
        else:
            documents_path = agent.run_extraction("https://example.com")
            chunks_path = agent.run_chunking(documents_path)
            embeddings_path = agent.run_embedding(chunks_path)
    else:
        # Run full pipeline
        try:
            documents_path = agent.run_extraction("https://example.com")
            chunks_path = agent.run_chunking(documents_path)
            embeddings_path = agent.run_embedding(chunks_path)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            if Path(default_chunks_path).exists():
                chunks_path = default_chunks_path
                embeddings_path = agent.run_embedding(chunks_path)
            else:
                sys.exit(1)

    # Interactive chat loop

    while True:
        query = input("Ask a question (or 'quit'): ")
        if query.lower() in ['quit', 'exit', 'q']:
            break

        if results := agent.run_search(query):
            # Generate response with context
            response = agent.run_chat(query, context=results)
        else:
            response = agent.run_chat(query)

