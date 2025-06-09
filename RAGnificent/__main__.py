"""
Command line entry point for RAGnificent.
"""

import sys
from pathlib import Path

# Use relative imports for internal modules
# Import fix applied
sys.path.insert(0, str(Path(__file__).parent.parent))

from RAGnificent.core.scraper import main

if __name__ == "__main__":
    # Re-execute main with the same arguments
    main(sys.argv[1:]) if len(sys.argv) > 1 else main([])
