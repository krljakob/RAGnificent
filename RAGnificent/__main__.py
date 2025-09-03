"""
Command line entry point for RAGnificent.
"""

import contextlib
import sys

from RAGnificent.core.config import get_config
from RAGnificent.core.scraper import main as scraper_main


def main():
    """Entry point function for console script."""
    # Configure logging explicitly (no side effects on import)
    with contextlib.suppress(Exception):
        get_config().configure_logging()
    # Re-execute main with the same arguments
    scraper_main(sys.argv[1:])


if __name__ == "__main__":
    main()
