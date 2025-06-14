#!/usr/bin/env python3
"""
Simple web UI for RAGnificent.

This provides a basic web interface for testing RAG functionality.
Note: This is a minimal implementation. For production use, consider FastAPI or Streamlit.
"""

import json
import logging
import os
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from RAGnificent.core.config import get_config
from RAGnificent.rag.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None


def initialize_pipeline():
    """Initialize the RAG pipeline."""
    global pipeline
    if pipeline is None:
        try:
            config = get_config()
            pipeline = Pipeline(collection_name=config.qdrant.collection)
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            pipeline = None
    return pipeline is not None


class RAGHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the RAG web UI."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path in {"/", "/index.html"}:
            self.serve_home_page()
        elif self.path == "/api/status":
            self.serve_status()
        else:
            self.send_error(404, "Page not found")

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/api/search":
            self.handle_search()
        elif self.path == "/api/scrape":
            self.handle_scrape()
        else:
            self.send_error(404, "Endpoint not found")

    def serve_home_page(self):
        """Serve the main HTML page."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>RAGnificent Web UI</title>
    <meta charset="utf-8">
    <style>
        :root {
            --bg-color: #1a1a1a;
            --text-color: #e0e0e0;
            --primary-color: #646cff;
            --primary-hover: #535bf2;
            --card-bg: #2a2a2a;
            --border-color: #404040;
            --success-color: #4caf50;
            --error-color: #f44336;
            --warning-color: #ff9800;
            --transition-speed: 0.2s;
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }

        .container {
            margin: 2rem 0;
        }

        .section {
            background-color: var(--card-bg);
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform var(--transition-speed) ease;
        }

        .section:hover {
            transform: translateY(-2px);
        }

        input[type="text"], textarea {
            width: 100%;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            background-color: var(--bg-color);
            color: var(--text-color);
            transition: border-color var(--transition-speed) ease;
        }

        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 2px rgba(100, 108, 255, 0.1);
        }

        button {
            background-color: var(--primary-color);
            color: white;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            margin: 0.5rem;
            font-weight: 500;
            transition: all var(--transition-speed) ease;
        }

        button:hover {
            background-color: var(--primary-hover);
            transform: translateY(-1px);
        }

        button:active {
            transform: translateY(0);
        }

        .result {
            background-color: var(--card-bg);
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 6px;
            border-left: 4px solid var(--primary-color);
        }

        .error {
            border-left-color: var(--error-color);
            background-color: rgba(244, 67, 54, 0.1);
        }

        .success {
            border-left-color: var(--success-color);
            background-color: rgba(76, 175, 80, 0.1);
        }

        .loading {
            color: #888;
            font-style: italic;
        }

        h1 {
            color: var(--text-color);
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }

        h2 {
            color: var(--primary-color);
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }

        .status {
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 6px;
            font-weight: 500;
        }

        .status.online {
            background-color: rgba(76, 175, 80, 0.1);
            border: 1px solid var(--success-color);
            color: var(--success-color);
        }

        .status.offline {
            background-color: rgba(244, 67, 54, 0.1);
            border: 1px solid var(--error-color);
            color: var(--error-color);
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg-color: #ffffff;
                --text-color: #213547;
                --card-bg: #f9f9f9;
                --border-color: #e0e0e0;
            }
        }
    </style>
</head>
<body>
    <h1>🔥 RAGnificent Web UI</h1>

    <div class="container">
        <div id="status" class="status">
            <strong>Status:</strong> <span id="status-text">Checking...</span>
        </div>
    </div>

    <div class="container">
        <div class="section">
            <h2>🔍 Search Documents</h2>
            <p>Search your indexed documents using semantic similarity.</p>
            <input type="text" id="searchQuery" placeholder="Enter your search query..." />
            <br>
            <button onclick="searchDocuments()">Search</button>
            <div id="searchResults"></div>
        </div>
    </div>

    <div class="container">
        <div class="section">
            <h2>🌐 Scrape & Index URL</h2>
            <p>Scrape a new URL and add it to the knowledge base.</p>
            <input type="text" id="scrapeUrl" placeholder="Enter URL to scrape..." />
            <br>
            <button onclick="scrapeUrl()">Scrape & Index</button>
            <div id="scrapeResults"></div>
        </div>
    </div>

    <script>
        // Check status on page load
        window.onload = function() {
            checkStatus();
        };

        function checkStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const statusDiv = document.getElementById('status');
                    const statusText = document.getElementById('status-text');

                    if (data.status === 'online') {
                        statusDiv.className = 'status online';
                        statusText.textContent = `Online - Collection: ${data.collection}`;
                    } else {
                        statusDiv.className = 'status offline';
                        statusText.textContent = 'Offline - Pipeline initialization failed';
                    }
                })
                .catch(error => {
                    const statusDiv = document.getElementById('status');
                    const statusText = document.getElementById('status-text');
                    statusDiv.className = 'status offline';
                    statusText.textContent = 'Error checking status';
                });
        }

        function searchDocuments() {
            const query = document.getElementById('searchQuery').value;
            const resultsDiv = document.getElementById('searchResults');

            if (!query.trim()) {
                resultsDiv.innerHTML = '<div class="result error">Please enter a search query.</div>';
                return;
            }

            resultsDiv.innerHTML = '<div class="result loading">Searching...</div>';

            fetch('/api/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultsDiv.innerHTML = `<div class="result error">Error: ${data.error}</div>`;
                    return;
                }

                if (data.results.length === 0) {
                    resultsDiv.innerHTML = '<div class="result">No results found.</div>';
                    return;
                }

                let html = `<h3>Found ${data.results.length} results:</h3>`;
                data.results.forEach((result, index) => {
                    html += `
                        <div class="result">
                            <strong>Result ${index + 1} (Score: ${result.score.toFixed(4)})</strong><br>
                            <strong>Source:</strong> ${result.metadata.document_url || 'Unknown'}<br>
                            <strong>Content:</strong> ${result.content.substring(0, 300)}...
                        </div>
                    `;
                });
                resultsDiv.innerHTML = html;
            })
            .catch(error => {
                resultsDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
            });
        }

        function scrapeUrl() {
            const url = document.getElementById('scrapeUrl').value;
            const resultsDiv = document.getElementById('scrapeResults');

            if (!url.trim()) {
                resultsDiv.innerHTML = '<div class="result error">Please enter a URL.</div>';
                return;
            }

            resultsDiv.innerHTML = '<div class="result loading">Scraping and indexing URL...</div>';

            fetch('/api/scrape', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url: url})
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    resultsDiv.innerHTML = `<div class="result error">Error: ${data.error}</div>`;
                    return;
                }

                resultsDiv.innerHTML = `
                    <div class="result success">
                        <strong>Successfully processed!</strong><br>
                        Documents: ${data.document_counts.documents || 0}<br>
                        Chunks: ${data.document_counts.chunks || 0}<br>
                        Embedded: ${data.document_counts.embedded_chunks || 0}<br>
                        Stored: ${data.document_counts.stored_vectors || 0}
                    </div>
                `;
            })
            .catch(error => {
                resultsDiv.innerHTML = `<div class="result error">Error: ${error.message}</div>`;
            });
        }

        // Allow Enter key to trigger search
        document.getElementById('searchQuery').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchDocuments();
            }
        });

        // Allow Enter key to trigger scrape
        document.getElementById('scrapeUrl').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                scrapeUrl();
            }
        });
    </script>
</body>
</html>
        """

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html_content.encode())

    def serve_status(self):
        """Serve status information."""
        if initialize_pipeline():
            config = get_config()
            response = {"status": "online", "collection": config.qdrant.collection}
        else:
            response = {"status": "offline", "error": "Pipeline initialization failed"}

        self.send_json_response(response)

    def handle_search(self):
        """Handle search requests."""
        if not initialize_pipeline():
            self.send_json_response({"error": "Pipeline not initialized"}, 500)
            return

        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            query = data.get("query", "").strip()
            if not query:
                self.send_json_response({"error": "Query is required"}, 400)
                return

            # Validate query length and content
            if len(query) > 1000:
                self.send_json_response({"error": "Query too long"}, 400)
                return

            # Basic sanitization
            import html

            query = html.escape(query)

            # Search documents
            results = pipeline.search_documents(query, limit=5, as_dict=True)

            response = {"query": query, "results": results}

            self.send_json_response(response)

        except Exception as e:
            logger.error(f"Search error: {e}")
            self.send_json_response({"error": str(e)}, 500)

    def handle_scrape(self):
        """Handle scrape requests."""
        if not initialize_pipeline():
            self.send_json_response({"error": "Pipeline not initialized"}, 500)
            return

        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            url = data.get("url", "").strip()
            if not url:
                self.send_json_response({"error": "URL is required"}, 400)
                return

            # Validate URL format and scheme
            from urllib.parse import urlparse

            try:
                parsed = urlparse(url)
                if parsed.scheme not in ["http", "https"]:
                    self.send_json_response({"error": "Invalid URL scheme"}, 400)
                    return
                if not parsed.netloc:
                    self.send_json_response({"error": "Invalid URL format"}, 400)
                    return
            except Exception:
                self.send_json_response({"error": "Invalid URL format"}, 400)
                return

            # Run the full pipeline
            result = pipeline.run_pipeline(
                url=url,
                run_extract=True,
                run_chunk=True,
                run_embed=True,
                run_store=True,
            )

            self.send_json_response(result)

        except Exception as e:
            logger.error(f"Scrape error: {e}")
            self.send_json_response({"error": str(e)}, 500)

    def send_json_response(self, data, status_code=200):
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        """Override to reduce log noise."""
        return


def main():
    """Run the web server."""
    port = int(os.environ.get("PORT", 8080))

    # Try to open browser automatically
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass  # Ignore if we can't open browser

    server = HTTPServer(("localhost", port), RAGHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
