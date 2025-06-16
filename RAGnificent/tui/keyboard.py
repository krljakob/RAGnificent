"""
Keyboard input handling for the RAGnificent TUI.

This module provides utilities for handling keyboard input in a non-blocking way
using a separate thread.
"""

import sys
import threading
import time
from typing import Callable, Optional


class KeyboardListener:
    """Listen for keyboard input in a separate thread.

    This class provides a context manager for handling keyboard input in a non-blocking
    way. It runs a background thread that reads input and calls the provided callback
    with each character received.

    Args:
        callback: A callable that will be called with each character received.
    """

    def __init__(self, callback: Callable[[str], None]):
        """Initialize the keyboard listener.

        Args:
            callback: Function to call with each keyboard input character.
        """
        self.callback = callback
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def __enter__(self):
        """Start the keyboard listener thread when entering the context."""
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the keyboard listener thread when exiting the context."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _listen(self):
        """Internal method that runs in a separate thread to listen for input."""
        while self.running:
            try:
                if char := sys.stdin.read(1):
                    self.callback(char)
            except (EOFError, KeyboardInterrupt):
                break
            except Exception:  # noqa: BLE001
                # Log any other exceptions and continue
                continue
            time.sleep(0.1)


def keyboard_listener(callback: Callable[[str], None]) -> KeyboardListener:
    """Create a keyboard listener context manager.

    This is a convenience function that creates and returns a new KeyboardListener
    instance configured with the provided callback.

    Args:
        callback: Function to call with each keyboard input character.

    Returns:
        A new KeyboardListener instance.
    """
    return KeyboardListener(callback)
