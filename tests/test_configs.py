"""Tests for hypersteer/utils/configs.py"""

import pytest
from omegaconf import OmegaConf

from hypersteer.utils.configs import (
    DatasetConfig,
    ModelConfig,
    TrainingArgs,
    WandbConfig,
    config_to_pydantic,
)


class TestConfigToPydantic:
    """Tests for config_to_pydantic function."""

    def test_converts_dictconfig_to_pydantic(self, sample_dictconfig):
        """Test converting DictConfig to Pydantic model."""
        result = config_to_pydantic(sample_dictconfig, WandbConfig)
        assert isinstance(result, WandbConfig)
        assert result.log is True
        assert result.project == "test-project"
        assert result.entity == "test-entity"

    def test_resolves_interpolations(self):
        """Test that OmegaConf interpolations are resolved."""
        cfg = OmegaConf.create(
            {
                "entity": "test",
                "project": "${entity}-project",
                "log": True,
            }
        )
        result = config_to_pydantic(cfg, WandbConfig)
        assert result.project == "test-project"


class TestWandbConfig:
    """Tests for WandbConfig defaults."""

    def test_default_values(self):
        """Test WandbConfig applies correct defaults."""
        config = WandbConfig()
        assert config.log is True
        assert config.log_code is True
        assert config.project is None
        assert config.entity is None

    def test_custom_values(self):
        """Test WandbConfig accepts custom values."""
        config = WandbConfig(
            log=False, project="my-project", tags=["tag1", "tag2"]
        )
        assert config.log is False
        assert config.project == "my-project"
        assert config.tags == ["tag1", "tag2"]


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test ModelConfig applies correct defaults."""
        config = ModelConfig()
        assert config.model_name == "HyperSteer"
        assert config.layer == 20
        assert config.hypernet_type == "regression"
        assert config.low_rank_dimension == 1

    def test_custom_model_name(self):
        """Test ModelConfig with custom model name."""
        config = ModelConfig(
            model_name="CustomModel",
            hypernet_type="attn",
            layer=15,
        )
        assert config.model_name == "CustomModel"
        assert config.hypernet_type == "attn"
        assert config.layer == 15


class TestTrainingArgs:
    """Tests for TrainingArgs."""

    def test_default_values(self):
        """Test TrainingArgs applies correct defaults."""
        config = TrainingArgs()
        assert config.batch_size == 16
        assert config.n_epochs == 3
        assert config.lr == 0.01
        assert config.optimizer == "adamw"

    def test_custom_training_values(self):
        """Test TrainingArgs with custom values."""
        config = TrainingArgs(batch_size=32, lr=1e-4, n_epochs=5)
        assert config.batch_size == 32
        assert config.lr == 1e-4
        assert config.n_epochs == 5


class TestDatasetConfig:
    """Tests for DatasetConfig with nested sub-configs."""

    def test_default_values(self):
        """Test DatasetConfig applies correct defaults."""
        config = DatasetConfig()
        assert config.dataset_type == "axbench"
        assert config.max_concepts == 500
        assert config.shuffle is True

    def test_nested_train_config(self):
        """Test DatasetConfig handles nested train sub-config."""
        config = DatasetConfig(
            train={
                "hf_dataset_name": "train-dataset",
                "hf_split": "train",
            }
        )
        assert config.train is not None
        assert config.train.hf_dataset_name == "train-dataset"
        assert config.train.hf_split == "train"

    def test_nested_eval_config(self):
        """Test DatasetConfig handles nested eval sub-config."""
        config = DatasetConfig(
            eval={
                "hf_dataset_name": "eval-dataset",
                "hf_split": "test",
            }
        )
        assert config.eval is not None
        assert config.eval.hf_dataset_name == "eval-dataset"
        assert config.eval.hf_split == "test"
