"""
Example demonstrating configuration management.

This example shows how to:
1. Load configuration from different file formats
2. Merge configurations from multiple sources
3. Override configuration settings programmatically
4. Save configuration to a file
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAGnificent.core.config import (
    AppConfig,
    ChunkingStrategy,
    EmbeddingModelType,
    load_config,
    load_configs_from_directory,
)


def main():
    """Run the configuration management example."""
    AppConfig()

    config_dir = project_root / "config" / "examples"
    yaml_config_path = config_dir / "production.yaml"

    load_config(yaml_config_path)

    json_config_path = config_dir / "local.json"

    load_config(json_config_path)

    load_configs_from_directory(config_dir)

    custom_config = AppConfig()

    custom_config.chunking.strategy = ChunkingStrategy.RECURSIVE
    custom_config.chunking.chunk_size = 500
    custom_config.chunking.chunk_overlap = 100

    custom_config.embedding.model_type = EmbeddingModelType.OPENAI
    custom_config.embedding.model_name = "text-embedding-3-small"

    output_dir = project_root / "data" / "config_example"
    os.makedirs(output_dir, exist_ok=True)

    yaml_output = output_dir / "custom_config.yaml"
    json_output = output_dir / "custom_config.json"

    custom_config.save_to_file(yaml_output, format="yaml")
    custom_config.save_to_file(json_output, format="json")

    load_config(yaml_output)


if __name__ == "__main__":
    main()
