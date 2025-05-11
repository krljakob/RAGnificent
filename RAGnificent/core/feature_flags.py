"""
Feature flags module for RAGnificent.

Provides a centralized system for managing feature flags,
allowing for gradual rollout of features and A/B testing.
"""

import logging
import os
import random
from enum import Enum
from typing import Any, Dict, Optional, Union

from RAGnificent.core.config import get_config

logger = logging.getLogger(__name__)


class FeatureFlag(str, Enum):
    """Enumeration of available feature flags."""

    ADVANCED_CHUNKING = "enable_advanced_chunking"
    PARALLEL_PROCESSING = "enable_parallel_processing"
    MEMORY_OPTIMIZATION = "enable_memory_optimization"
    CACHING = "enable_caching"
    BENCHMARKING = "enable_benchmarking"
    SECURITY_FEATURES = "enable_security_features"

    EXPERIMENTAL_EMBEDDINGS = "enable_experimental_embeddings"
    HYBRID_SEARCH = "enable_hybrid_search"
    STREAMING_RESPONSES = "enable_streaming_responses"
    ADAPTIVE_CHUNKING = "enable_adaptive_chunking"


class FeatureFlagManager:
    """
    Manager for feature flags.

    Provides methods for checking if features are enabled,
    and for enabling/disabling features at runtime.
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature flag manager.

        Args:
            config_dict: Optional dictionary with feature flag configuration
        """
        self.flags: Dict[str, bool] = {
            FeatureFlag.ADVANCED_CHUNKING.value: True,
            FeatureFlag.PARALLEL_PROCESSING.value: True,
            FeatureFlag.MEMORY_OPTIMIZATION.value: True,
            FeatureFlag.CACHING.value: True,
            FeatureFlag.BENCHMARKING.value: False,
            FeatureFlag.SECURITY_FEATURES.value: True,
            FeatureFlag.EXPERIMENTAL_EMBEDDINGS.value: False,
            FeatureFlag.HYBRID_SEARCH.value: False,
            FeatureFlag.STREAMING_RESPONSES.value: False,
            FeatureFlag.ADAPTIVE_CHUNKING.value: False,
        }

        self._load_from_env()

        if config_dict:
            self._load_from_dict(config_dict)
        else:
            try:
                config = get_config()
                if hasattr(config, "features"):
                    self._load_from_dict(config.features)
            except Exception as e:
                logger.warning(f"Failed to load feature flags from config: {e}")

        self._log_enabled_features()

    def _load_from_env(self) -> None:
        """Load feature flags from environment variables."""
        prefix = "RAGNIFICENT_FEATURE_"

        for flag in FeatureFlag:
            env_var = f"{prefix}{flag.name}"
            if env_var in os.environ:
                value = os.environ[env_var].lower()
                self.flags[flag.value] = value in ("1", "true", "yes", "on")

    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load feature flags from configuration dictionary.

        Args:
            config_dict: Dictionary with feature flag configuration
        """
        for flag_name, flag_value in config_dict.items():
            if flag_name in self.flags:
                self.flags[flag_name] = bool(flag_value)

    def _log_enabled_features(self) -> None:
        """Log all enabled features."""
        enabled_features = [name for name, enabled in self.flags.items() if enabled]
        logger.info(f"Enabled features: {', '.join(enabled_features)}")

    def is_enabled(self, feature: Union[str, FeatureFlag]) -> bool:
        """
        Check if a feature is enabled.

        Args:
            feature: Feature flag name or enum value

        Returns:
            True if the feature is enabled, False otherwise
        """
        feature_name = feature.value if isinstance(feature, FeatureFlag) else feature
        return self.flags.get(feature_name, False)

    def enable(self, feature: Union[str, FeatureFlag]) -> None:
        """
        Enable a feature.

        Args:
            feature: Feature flag name or enum value
        """
        feature_name = feature.value if isinstance(feature, FeatureFlag) else feature
        if feature_name in self.flags:
            self.flags[feature_name] = True
            logger.info(f"Feature enabled: {feature_name}")

    def disable(self, feature: Union[str, FeatureFlag]) -> None:
        """
        Disable a feature.

        Args:
            feature: Feature flag name or enum value
        """
        feature_name = feature.value if isinstance(feature, FeatureFlag) else feature
        if feature_name in self.flags:
            self.flags[feature_name] = False
            logger.info(f"Feature disabled: {feature_name}")

    def toggle(self, feature: Union[str, FeatureFlag]) -> bool:
        """
        Toggle a feature.

        Args:
            feature: Feature flag name or enum value

        Returns:
            New state of the feature (True if enabled, False if disabled)
        """
        feature_name = feature.value if isinstance(feature, FeatureFlag) else feature
        if feature_name in self.flags:
            self.flags[feature_name] = not self.flags[feature_name]
            state = "enabled" if self.flags[feature_name] else "disabled"
            logger.info(f"Feature {feature_name} {state}")
            return self.flags[feature_name]

        return False

    def get_all_flags(self) -> Dict[str, bool]:
        """
        Get all feature flags and their states.

        Returns:
            Dictionary mapping feature names to their states
        """
        return dict(self.flags)

    def reset_to_defaults(self) -> None:
        """Reset all feature flags to their default values."""
        self.__init__()
        logger.info("Feature flags reset to defaults")


_FEATURE_FLAG_MANAGER = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """
    Get the singleton feature flag manager instance.

    Returns:
        FeatureFlagManager instance
    """
    global _FEATURE_FLAG_MANAGER
    if _FEATURE_FLAG_MANAGER is None:
        _FEATURE_FLAG_MANAGER = FeatureFlagManager()
    return _FEATURE_FLAG_MANAGER


def is_feature_enabled(feature: Union[str, FeatureFlag]) -> bool:
    """
    Check if a feature is enabled.

    This is a convenience function that uses the singleton manager.

    Args:
        feature: Feature flag name or enum value

    Returns:
        True if the feature is enabled, False otherwise
    """
    return get_feature_flag_manager().is_enabled(feature)


class FeatureGate:
    """
    Feature gate for controlling access to features.

    This class provides a way to gradually roll out features
    to a percentage of users or based on other criteria.
    """

    def __init__(
        self,
        feature: Union[str, FeatureFlag],
        rollout_percentage: float = 100.0,
        user_id_salt: str = "ragnificent",
    ):
        """
        Initialize the feature gate.

        Args:
            feature: Feature flag name or enum value
            rollout_percentage: Percentage of users to enable the feature for (0-100)
            user_id_salt: Salt to use for user ID hashing
        """
        self.feature = feature
        self.rollout_percentage = max(0.0, min(100.0, rollout_percentage))
        self.user_id_salt = user_id_salt

    def is_enabled(self, user_id: Optional[str] = None) -> bool:
        """
        Check if the feature is enabled for a specific user.

        Args:
            user_id: Optional user identifier for percentage-based rollout

        Returns:
            True if the feature is enabled, False otherwise
        """
        if not is_feature_enabled(self.feature):
            return False

        if self.rollout_percentage >= 100.0:
            return True

        if self.rollout_percentage <= 0.0:
            return False

        if user_id is None:
            return random.random() * 100.0 < self.rollout_percentage

        import hashlib

        hash_input = f"{self.user_id_salt}:{user_id}:{self.feature}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        user_percentage = (hash_value % 10000) / 100.0

        return user_percentage < self.rollout_percentage
