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
    load_configs_from_directory
)

def main():
    """Run the configuration management example."""
    print("Loading default configuration...")
    default_config = AppConfig()
    
    print(f"Default chunking strategy: {default_config.chunking.strategy}")
    print(f"Default embedding model: {default_config.embedding.model_name}")
    
    config_dir = project_root / "config" / "examples"
    yaml_config_path = config_dir / "production.yaml"
    
    print("\nLoading configuration from YAML file...")
    yaml_config = load_config(yaml_config_path)
    
    print(f"YAML chunking strategy: {yaml_config.chunking.strategy}")
    print(f"YAML embedding model: {yaml_config.embedding.model_name}")
    
    json_config_path = config_dir / "local.json"
    
    print("\nLoading configuration from JSON file...")
    json_config = load_config(json_config_path)
    
    print(f"JSON chunking strategy: {json_config.chunking.strategy}")
    print(f"JSON embedding model: {json_config.embedding.model_name}")
    
    print("\nLoading and merging configurations from directory...")
    merged_config = load_configs_from_directory(config_dir)
    
    print(f"Merged chunking strategy: {merged_config.chunking.strategy}")
    print(f"Merged embedding model: {merged_config.embedding.model_name}")
    
    print("\nOverriding configuration programmatically...")
    custom_config = AppConfig()
    
    custom_config.chunking.strategy = ChunkingStrategy.RECURSIVE
    custom_config.chunking.chunk_size = 500
    custom_config.chunking.chunk_overlap = 100
    
    custom_config.embedding.model_type = EmbeddingModelType.OPENAI
    custom_config.embedding.model_name = "text-embedding-3-small"
    
    print(f"Custom chunking strategy: {custom_config.chunking.strategy}")
    print(f"Custom embedding model: {custom_config.embedding.model_name}")
    
    output_dir = project_root / "data" / "config_example"
    os.makedirs(output_dir, exist_ok=True)
    
    yaml_output = output_dir / "custom_config.yaml"
    json_output = output_dir / "custom_config.json"
    
    print("\nSaving configuration to files...")
    custom_config.save_to_file(yaml_output, format="yaml")
    custom_config.save_to_file(json_output, format="json")
    
    print(f"Saved YAML configuration to {yaml_output}")
    print(f"Saved JSON configuration to {json_output}")
    
    print("\nLoading saved configuration...")
    loaded_config = load_config(yaml_output)
    
    print(f"Loaded chunking strategy: {loaded_config.chunking.strategy}")
    print(f"Loaded embedding model: {loaded_config.embedding.model_name}")

if __name__ == "__main__":
    main()
