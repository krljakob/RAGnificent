#!/usr/bin/env python3
"""
Legacy command line interface for RAGnificent.

Note: For better compatibility with the package structure, consider using:
    python -m RAGnificent [arguments]
instead of:
    python main.py [arguments]
"""

import sys

from RAGnificent.core.scraper import main

if __name__ == "__main__":
    # Re-execute main with the same arguments
    main(sys.argv[1:]) if len(sys.argv) > 1 else main([])
