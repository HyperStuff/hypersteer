from .base import BaseModel
from .hypersteer_regression import HyperSteerRegression
from .model import Model
from .prompting import PromptDetection, PromptSteering, SimplePromptSteering

MODELS = {
    "HyperSteerRegression": HyperSteerRegression,
    "PromptSteering": PromptSteering,
    "SimplePromptSteering": SimplePromptSteering,
    "PromptDetection": PromptDetection,
}


def get_model(model_type: str, **kwargs):
    """
    Get a model instance by type.

    Args:
        model_type: Model name (e.g., 'HyperSteer', 'HyperSteerAttn', 'HyperSteerRegression')
        **kwargs: Arguments to pass to the model constructor

    Returns:
        An instance of the requested model
    """
    # Handle "HyperSteer" alias - dispatch based on hypernet_type in model_config
    if model_type == "HyperSteer":
        model_config = kwargs.get("model_config")
        if model_config and hasattr(model_config, "hypernet_type"):
            hypernet_type = model_config.hypernet_type
            if hypernet_type == "regression":
                model_type = "HyperSteerRegression"
            else:
                model_type = "HyperSteerAttn"

    if model_type not in MODELS:
        raise ValueError(
            f"Unknown model: {model_type}. Available: {list(MODELS.keys())}"
        )

    return MODELS[model_type](**kwargs)


def list_available_models() -> list[str]:
    """List all available model types."""
    return list(MODELS.keys())


__all__ = [
    "get_model",
    "list_available_models",
    "BaseModel",
    "Model",
    "HyperSteerAttn",
    "HyperSteerRegression",
    "PromptSteering",
    "SimplePromptSteering",
    "PromptDetection",
]
