import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import requests

class DocumentConverter:
    def __init__(self):
        self.documents = []
        self.logger = logging.getLogger(__name__)

    def convert(self, url: str) -> Optional[Dict[str, Any]]:
        """Convert a web page URL into a standardized document format.
        
        Args:
            url: URL of the web page to convert
            
        Returns:
            Dictionary containing document metadata and text content
        """
        try:
            # Fetch the web page
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else url
            
            # Extract main content - this is a simple implementation
            # Could be enhanced with more sophisticated content extraction
            main_content = soup.find('main') or soup.find('article') or soup.body
            text = main_content.get_text(separator='\n', strip=True) if main_content else ""
            
            # Create document structure
            document = {
                'id': str(hash(url)),
                'url': url,
                'title': title,
                'text': text,
                'source': url
            }
            
            self.documents.append(document)
            return document
            
        except Exception as e:
            self.logger.error(f"Error converting {url}: {str(e)}")
            return None

    def save(self, path: str) -> None:
        """Save converted documents to a JSON file.
        
        Args:
            path: Path to save the documents to
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Saved {len(self.documents)} documents to {path}")

    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all converted documents.
        
        Returns:
            List of converted documents
        """
        return self.documents
