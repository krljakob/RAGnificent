from abc import ABC, abstractmethod
from typing import Any, Dict


class StatsMixin:
    """Mixin class for statistics collection functionality."""

    def __init__(self, *args, enable_stats: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_stats = enable_stats

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the component.

        Returns:
            Dictionary of component statistics
        """
        if not self.enable_stats:
            return {"stats_disabled": True}
        return self._get_stats_implementation()

    def _get_stats_implementation(self) -> Dict[str, Any]:
        """Class-specific stats implementation."""
        return {}
