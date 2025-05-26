from .base import ModelRegistry


def get_model(model_type: str, **kwargs):
    """
    Get a model instance by type.

    Args:
        model_type: The type of model to create (e.g., 'HyperSteer', 'PromptSteering')
        **kwargs: Arguments to pass to the model constructor

    Returns:
        An instance of the requested model
    """
    model_class = ModelRegistry.get_model(model_type)
    return model_class(**kwargs)


def list_available_models():
    """
    List all available model types.

    Returns:
        List of available model type names
    """
    return ModelRegistry.list_models()
