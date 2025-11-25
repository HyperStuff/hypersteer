"""Shared fixtures for hypersteer tests."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer with essential attributes."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 1

    def tokenize(text, **kwargs):
        # Simple mock: return tensor with length proportional to text
        length = min(len(text.split()), 10)
        return {"input_ids": torch.tensor([[i for i in range(length)]])}

    tokenizer.side_effect = tokenize
    tokenizer.__call__ = tokenize
    return tokenizer


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for data module tests."""
    return pd.DataFrame(
        {
            "concept_id": [0, 1, 2],
            "input_concept": ["formal", "humorous", "technical"],
            "output_concept": ["formal response", "funny response", "technical response"],
            "input": ["Hello", "Hi there", "Greetings"],
            "output": [" world", " friend", " colleague"],
        }
    )


@pytest.fixture
def sample_dictconfig():
    """Create a sample OmegaConf DictConfig for config tests."""
    return OmegaConf.create(
        {
            "log": True,
            "project": "test-project",
            "entity": "test-entity",
            "run_name": "test-run",
        }
    )


@pytest.fixture
def model_config_dict():
    """Create a sample model config dictionary."""
    return {
        "model_name": "HyperSteer",
        "target_model_name": "google/gemma-2-2b-it",
        "base_model_name": "google/gemma-2-2b",
        "layer": 20,
        "hypernet_type": "regression",
    }
