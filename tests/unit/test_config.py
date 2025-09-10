import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

try:
    from RAGnificent.core.config import (
        AppConfig,
        ChunkingStrategy,
        EmbeddingModelType,
        load_config,
        load_configs_from_directory,
    )
except ImportError:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from RAGnificent.core.config import (
        AppConfig,
        ChunkingStrategy,
        EmbeddingModelType,
        load_config,
        load_configs_from_directory,
    )


class TestConfigFileSupport(unittest.TestCase):
    """Test the enhanced configuration file support."""

    def test_app_config_initialization_with_dict_override(self):
        """Ensure that AppConfig applies overrides from config_dict"""
        override = {
            "chunking": {"strategy": ChunkingStrategy.SEMANTIC.value},
            "embedding": {"model_type": EmbeddingModelType.OPENAI.value},
        }
        config = AppConfig(config_dict=override)
        # overrides should map back to enum values
        self.assertEqual(config.chunking.strategy, ChunkingStrategy.SEMANTIC)
        self.assertEqual(config.embedding.model_type, EmbeddingModelType.OPENAI)

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        yaml_config = {
            "chunking": {
                "strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 100,
            },
            "embedding": {
                "model_type": "openai",
                "model_name": "text-embedding-3-small",
            },
            "data_dir": "/custom/data/path",
        }

        yaml_path = self.config_dir / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_config, f)

        config = load_config(yaml_path)

        self.assertEqual(config.chunking.strategy, ChunkingStrategy.RECURSIVE)
        self.assertEqual(config.chunking.chunk_size, 500)
        self.assertEqual(config.chunking.chunk_overlap, 100)
        self.assertEqual(config.embedding.model_type, EmbeddingModelType.OPENAI)
        self.assertEqual(config.embedding.model_name, "text-embedding-3-small")
        self.assertEqual(str(config.data_dir), "/custom/data/path")

    def test_load_json_config(self):
        """Test loading configuration from JSON file."""
        json_config = {
            "scraper": {
                "max_depth": 3,
                "timeout": 20,
                "use_rust_implementation": False,
            },
            "search": {
                "max_results": 10,
                "score_threshold": 0.8,
            },
            "models_dir": "/custom/models/path",
        }

        json_path = self.config_dir / "test_config.json"
        with open(json_path, "w") as f:
            json.dump(json_config, f)

        config = load_config(json_path)

        self.assertEqual(config.scraper.max_depth, 3)
        self.assertEqual(config.scraper.timeout, 20)
        self.assertEqual(config.scraper.use_rust_implementation, False)
        self.assertEqual(config.search.max_results, 10)
        self.assertEqual(config.search.score_threshold, 0.8)
        self.assertEqual(str(config.models_dir), "/custom/models/path")

    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = AppConfig()

        config.chunking.strategy = ChunkingStrategy.RECURSIVE
        config.chunking.chunk_size = 750
        config.embedding.model_type = EmbeddingModelType.OPENAI

        yaml_path = self.config_dir / "saved_config.yaml"
        config.save_to_file(yaml_path)

        json_path = self.config_dir / "saved_config.json"
        config.save_to_file(json_path, format="json")

        self.assertTrue(yaml_path.exists())
        self.assertTrue(json_path.exists())

        yaml_config = load_config(yaml_path)
        json_config = load_config(json_path)

        self.assertEqual(yaml_config.chunking.strategy, ChunkingStrategy.RECURSIVE)
        self.assertEqual(yaml_config.chunking.chunk_size, 750)
        self.assertEqual(yaml_config.embedding.model_type, EmbeddingModelType.OPENAI)

        self.assertEqual(json_config.chunking.strategy, ChunkingStrategy.RECURSIVE)
        self.assertEqual(json_config.chunking.chunk_size, 750)
        self.assertEqual(json_config.embedding.model_type, EmbeddingModelType.OPENAI)

    def test_load_configs_from_directory(self):
        """Test loading and merging multiple configuration files."""
        base_config = {
            "chunking": {
                "strategy": "sliding_window",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
            "embedding": {
                "model_type": "sentence_transformer",
                "model_name": "base-model",
            },
            "data_dir": "/base/data/path",
        }

        override_config = {
            "chunking": {
                "chunk_size": 500,
            },
            "embedding": {
                "model_name": "override-model",
            },
            "models_dir": "/override/models/path",
        }

        base_path = self.config_dir / "01_base.yaml"
        override_path = self.config_dir / "02_override.json"

        with open(base_path, "w") as f:
            yaml.dump(base_config, f)

        with open(override_path, "w") as f:
            json.dump(override_config, f)

        config = load_configs_from_directory(self.config_dir)

        self.assertEqual(config.chunking.strategy, ChunkingStrategy.SEMANTIC)
        self.assertEqual(config.chunking.chunk_size, 500)  # Overridden
        self.assertEqual(config.chunking.chunk_overlap, 200)  # From base
        self.assertEqual(
            config.embedding.model_type, EmbeddingModelType.SENTENCE_TRANSFORMER
        )
        self.assertEqual(config.embedding.model_name, "override-model")  # Overridden
        self.assertEqual(str(config.data_dir), "/base/data/path")
        self.assertEqual(str(config.models_dir), "/override/models/path")


if __name__ == "__main__":
    unittest.main()
