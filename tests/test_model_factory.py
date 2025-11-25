"""Tests for hypersteer/models/__init__.py model factory."""

from unittest.mock import MagicMock, patch

import pytest

from hypersteer.models import MODELS, get_model, list_available_models


class TestListAvailableModels:
    """Tests for list_available_models function."""

    def test_returns_list(self):
        """Test list_available_models returns a list."""
        models = list_available_models()
        assert isinstance(models, list)

    def test_contains_expected_models(self):
        """Test list contains expected model names."""
        models = list_available_models()
        expected = ["HyperSteer", "HyperSteerAttn", "HyperSteerRegression", "PromptSteering"]
        for model_name in expected:
            assert model_name in models


class TestGetModel:
    """Tests for get_model function."""

    def test_raises_for_unknown_model(self):
        """Test get_model raises ValueError for unknown model type."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("NonExistentModel")

    def test_dispatches_hypersteer_regression(self):
        """Test HyperSteer dispatches to HyperSteerRegression when hypernet_type='regression'."""
        mock_config = MagicMock()
        mock_config.hypernet_type = "regression"

        with patch.dict(MODELS, {"HyperSteerRegression": MagicMock(return_value="regression_model")}):
            result = get_model("HyperSteer", model_config=mock_config)
            assert result == "regression_model"

    def test_dispatches_hypersteer_attn(self):
        """Test HyperSteer dispatches to HyperSteerAttn when hypernet_type='attn'."""
        mock_config = MagicMock()
        mock_config.hypernet_type = "attn"

        with patch.dict(MODELS, {"HyperSteerAttn": MagicMock(return_value="attn_model")}):
            result = get_model("HyperSteer", model_config=mock_config)
            assert result == "attn_model"

    def test_dispatches_without_model_config(self):
        """Test HyperSteer without model_config defaults to HyperSteerAttn."""
        with patch.dict(MODELS, {"HyperSteer": MagicMock(return_value="default_model"), "HyperSteerAttn": MagicMock(return_value="default_model")}):
            # When no model_config provided, HyperSteer stays as HyperSteer (defaults to HyperSteerAttn class)
            result = get_model("HyperSteerAttn")
            assert result == "default_model"


class TestModelsRegistry:
    """Tests for MODELS registry."""

    def test_registry_is_dict(self):
        """Test MODELS is a dictionary."""
        assert isinstance(MODELS, dict)

    def test_hypersteer_alias(self):
        """Test HyperSteer is aliased to HyperSteerAttn."""
        assert MODELS["HyperSteer"] == MODELS["HyperSteerAttn"]
