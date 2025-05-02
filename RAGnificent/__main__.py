"""
Command line entry point for RAGnificent.
"""

import sys

from RAGnificent.core.scraper import main

if __name__ == "__main__":
    # Re-execute main with the same arguments
    main(sys.argv[1:]) if len(sys.argv) > 1 else main([])
