#!/usr/bin/env python3
"""
RAGnificent Comprehensive Demo
Demonstrates end-to-end scraping, chunking, embedding, and search functionality.
Consolidates functionality from multiple demo scripts.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Set up environment
sys.path.insert(0, str(Path(__file__).parent.parent))

from RAGnificent.core.config import EmbeddingModelType
from RAGnificent.core.scraper import MarkdownScraper
from RAGnificent.rag.embedding import EmbeddingService
from RAGnificent.rag.pipeline import Pipeline
from RAGnificent.rag.vector_store import VectorStore


class RAGDemo:
    """Unified RAG demonstration class"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.scraper = MarkdownScraper(
            requests_per_second=1.0, chunk_size=1000, chunk_overlap=200
        )
        self.embedding_service = EmbeddingService(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMER,
            model_name=model_name,
        )
        
    def print_step(self, step: str, description: str = ""):
        """Print a formatted step header"""
        print(f"\n{'='*60}")
        print(f"STEP {step}")
        if description:
            print(f"{description}")
        print(f"{'='*60}")
        
    def print_results(self, title: str, items: List[Any], limit: int = 3):
        """Print formatted results"""
        print(f"\n{title}:")
        print("-" * 40)
        for i, item in enumerate(items[:limit]):
            try:
                if hasattr(item, "to_dict"):
                    item_dict = item.to_dict()
                    content = item_dict.get("content", "")[:150]
                    print(f"{i+1}. {content}...")
                elif isinstance(item, dict):
                    content = item.get("content", "")[:150]
                    print(f"{i+1}. {content}...")
                else:
                    print(f"{i+1}. {str(item)[:150]}...")
            except Exception as e:
                print(f"{i+1}. Error displaying item: {e}")
        if len(items) > limit:
            print(f"... and {len(items) - limit} more items")

    def scrape_and_chunk(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape URLs and create chunks"""
        self.print_step("1", "SCRAPING & CHUNKING - Converting web content to structured chunks")
        
        all_chunks = []
        
        for i, url in enumerate(urls, 1):
            print(f"\nProcessing URL {i}/{len(urls)}: {url}")
            
            try:
                start_time = time.time()
                html_content = self.scraper.scrape_website(url)
                markdown_content = self.scraper.convert_to_markdown(html_content, url)
                scrape_time = time.time() - start_time
                
                start_time = time.time()
                chunks = self.scraper.create_chunks(markdown_content, url)
                chunk_time = time.time() - start_time
                
                all_chunks.extend(chunks)
                
                print(f"  ✓ Scraped in {scrape_time:.2f}s, chunked in {chunk_time:.2f}s")
                print(f"  ✓ Created {len(chunks)} chunks")
                
            except Exception as e:
                print(f"  ✗ Error processing {url}: {e}")
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        if all_chunks:
            self.print_results("Sample Chunks", all_chunks, 2)
            
        return all_chunks

    def generate_embeddings(self, chunks: List[Dict[str, Any]], limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks"""
        self.print_step("2", "EMBEDDING GENERATION - Converting text to vector embeddings")
        
        if limit:
            chunks = chunks[:limit]
            print(f"Processing first {limit} chunks for demo")
        
        embedded_chunks = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            try:
                embedded_chunk = self.embedding_service.embed_chunk(chunk)
                embedded_chunks.append(embedded_chunk)
                
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"  ✓ Processed {i+1}/{len(chunks)} chunks ({rate:.1f} chunks/sec)")
                    
            except Exception as e:
                print(f"  ✗ Error embedding chunk {i}: {e}")
        
        total_time = time.time() - start_time
        print(f"\nEmbedding generation completed in {total_time:.2f}s")
        print(f"Successfully embedded {len(embedded_chunks)}/{len(chunks)} chunks")
        
        if embedded_chunks:
            embedding_dim = len(embedded_chunks[0].get("embedding", []))
            print(f"Embedding dimension: {embedding_dim}")
            
        return embedded_chunks

    def store_in_vector_db(self, embedded_chunks: List[Dict[str, Any]]) -> Optional[VectorStore]:
        """Store embeddings in vector database"""
        self.print_step("3", "VECTOR STORAGE - Storing embeddings in vector database")
        
        try:
            vector_store = VectorStore(host=":memory:", collection_name="ragnificent_demo")
            
            if embedded_chunks:
                vector_store.store_documents(embedded_chunks)
                info = vector_store.get_collection_info()
                print(f"✓ Stored {len(embedded_chunks)} documents")
                print(f"✓ Collection info: {info}")
            else:
                print("✗ No embedded chunks to store")
                
            return vector_store
            
        except Exception as e:
            print(f"✗ Error setting up vector storage: {e}")
            return None

    def demonstrate_search(self, vector_store: Optional[VectorStore], embedded_chunks: List[Dict[str, Any]]):
        """Demonstrate search functionality"""
        self.print_step("4", "SEARCH & RETRIEVAL - Semantic search over stored content")
        
        test_queries = [
            "How do I use Python as a calculator?",
            "What are Python strings?",
            "How to define variables in Python?",
            "Python comments and syntax",
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            try:
                if vector_store:
                    # Use vector store search
                    results = vector_store.search(query=query, limit=3, threshold=0.3)
                    if results:
                        for i, result in enumerate(results, 1):
                            score = result.get("score", 0)
                            content = result.get("content", "")[:100]
                            source = result.get("source_url", "Unknown")
                            print(f"  {i}. Score: {score:.3f} | {content}... | Source: {source}")
                    else:
                        print("  No results found")
                else:
                    # Fallback to manual cosine similarity
                    results = self._manual_search(query, embedded_chunks, top_k=3)
                    for i, (similarity, chunk) in enumerate(results, 1):
                        content = chunk.get("content", "")[:100]
                        print(f"  {i}. Score: {similarity:.3f} | {content}...")
                        
            except Exception as e:
                print(f"  ✗ Search error: {e}")

    def _manual_search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 3):
        """Manual cosine similarity search"""
        query_result = self.embedding_service.embed_chunk(query)
        query_embedding = np.array(query_result["embedding"])
        
        similarities = []
        for chunk in chunks:
            if chunk.get("embedding"):
                chunk_embedding = np.array(chunk["embedding"])
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append((similarity, chunk))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]

    def demonstrate_pipeline(self):
        """Demonstrate end-to-end pipeline"""
        self.print_step("5", "END-TO-END PIPELINE - Complete RAG workflow using Pipeline class")
        
        try:
            pipeline = Pipeline(
                collection_name="ragnificent_e2e", 
                embedding_model=self.model_name
            )
            
            test_url = "https://docs.python.org/3/tutorial/introduction.html"
            print(f"Processing URL: {test_url}")
            
            if pipeline.process_url(test_url):
                print("✓ URL processed successfully")
                
                query = "What is Python?"
                results = pipeline.search(query, limit=3)
                
                print(f"\nSearch results for '{query}':")
                for i, result in enumerate(results, 1):
                    content = str(result)[:100] if result else "No content"
                    print(f"  {i}. {content}...")
            else:
                print("✗ Failed to process URL")
                
        except Exception as e:
            print(f"✗ Pipeline error: {e}")

    def save_results(self, chunks: List[Dict[str, Any]], embedded_chunks: List[Dict[str, Any]], 
                    output_file: str = "rag_demo_results.jsonl"):
        """Save processing results"""
        print(f"\nSaving results to {output_file}")
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for chunk in embedded_chunks:
                    # Ensure embedding is serializable
                    chunk_data = dict(chunk)
                    if "embedding" in chunk_data and hasattr(chunk_data["embedding"], "tolist"):
                        chunk_data["embedding"] = chunk_data["embedding"].tolist()
                    f.write(json.dumps(chunk_data) + "\n")
            
            file_size = Path(output_file).stat().st_size / (1024 * 1024)
            print(f"✓ Saved {len(embedded_chunks)} chunks ({file_size:.2f} MB)")
            
        except Exception as e:
            print(f"✗ Error saving results: {e}")

    def load_existing_chunks(self, chunk_files: List[str]) -> List[Dict[str, Any]]:
        """Load chunks from existing JSONL files"""
        self.print_step("LOAD", "Loading existing chunks from files")
        
        all_chunks = []
        for chunk_file in chunk_files:
            if Path(chunk_file).exists():
                print(f"Loading {chunk_file}")
                with open(chunk_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            chunk = json.loads(line.strip())
                            all_chunks.append(chunk)
                        except Exception as e:
                            print(f"Error loading chunk: {e}")
            else:
                print(f"File not found: {chunk_file}")
        
        print(f"Loaded {len(all_chunks)} chunks total")
        return all_chunks


def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(description="RAGnificent Comprehensive Demo")
    parser.add_argument("--mode", choices=["full", "existing", "pipeline", "basic"], 
                       default="basic", help="Demo mode to run")
    parser.add_argument("--urls", nargs="*", 
                       default=["https://docs.python.org/3/tutorial/introduction.html",
                               "https://docs.python.org/3/tutorial/controlflow.html"],
                       help="URLs to process")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", 
                       help="Embedding model to use")
    parser.add_argument("--limit", type=int, help="Limit number of chunks to process")
    parser.add_argument("--chunk-files", nargs="*", 
                       default=["tutorial_chunks/chunks.jsonl", 
                               "demo_chunks/3_tutorial_introduction/chunks.jsonl"],
                       help="Existing chunk files to load")
    parser.add_argument("--output", default="rag_demo_results.jsonl",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    print("RAGnificent Comprehensive Demo")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    
    demo = RAGDemo(model_name=args.model)
    start_time = time.time()
    
    try:
        if args.mode == "full":
            # Full workflow: scrape, embed, store, search
            chunks = demo.scrape_and_chunk(args.urls)
            embedded_chunks = demo.generate_embeddings(chunks, limit=args.limit)
            vector_store = demo.store_in_vector_db(embedded_chunks)
            demo.demonstrate_search(vector_store, embedded_chunks)
            demo.demonstrate_pipeline()
            demo.save_results(chunks, embedded_chunks, args.output)
            
        elif args.mode == "existing":
            # Use existing chunks
            chunks = demo.load_existing_chunks(args.chunk_files)
            if chunks:
                embedded_chunks = demo.generate_embeddings(chunks, limit=args.limit)
                vector_store = demo.store_in_vector_db(embedded_chunks)
                demo.demonstrate_search(vector_store, embedded_chunks)
                demo.save_results(chunks, embedded_chunks, args.output)
            else:
                print("No existing chunks found")
                
        elif args.mode == "pipeline":
            # Pipeline demo only
            demo.demonstrate_pipeline()
            
        elif args.mode == "basic":
            # Basic functionality demo
            chunks = demo.scrape_and_chunk(args.urls[:1])  # Just first URL
            if chunks:
                embedded_chunks = demo.generate_embeddings(chunks[:5])  # Limit to 5
                demo.demonstrate_search(None, embedded_chunks)
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Demo completed in {elapsed:.2f} seconds")
        print(f"{'='*60}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        import traceback
        print(f"\nDemo failed with error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()