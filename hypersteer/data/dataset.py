from . import axbench  # Import to register the factories
from .base import DatasetFactoryRegistry, TrainingDatasetRegistry


def get_dataset_factory(factory_type: str, **kwargs):
    """
    Get a dataset factory instance by type.

    Args:
        factory_type: The type of factory to create (e.g., 'axbench')
        **kwargs: Arguments to pass to the factory constructor

    Returns:
        An instance of the requested dataset factory
    """
    factory_class = DatasetFactoryRegistry.get_factory(factory_type)
    return factory_class(**kwargs)


def get_steering_dataset_factory(factory_type: str, **kwargs):
    """
    Get a steering dataset factory instance by type.

    Args:
        factory_type: The type of factory to create (e.g., 'axbench')
        **kwargs: Arguments to pass to the factory constructor

    Returns:
        An instance of the requested steering dataset factory
    """
    # Automatically append '_steering' suffix if not present
    if not factory_type.endswith("_steering"):
        factory_type = f"{factory_type}_steering"

    factory_class = DatasetFactoryRegistry.get_factory(factory_type)
    return factory_class(**kwargs)


def get_training_dataset(dataset_type: str, **kwargs):
    """
    Get a training dataset using a registered training dataset function.

    Args:
        dataset_type: The type of training dataset to create (e.g., 'axbench')
        **kwargs: Arguments to pass to the training dataset function

    Returns:
        A processed HuggingFace Dataset ready for training
    """
    training_function = TrainingDatasetRegistry.get_function(dataset_type)
    return training_function(**kwargs)


def list_available_factories():
    """
    List all available factory types.

    Returns:
        List of available factory type names
    """
    return DatasetFactoryRegistry.list_factories()


def list_available_training_datasets():
    """
    List all available training dataset function types.

    Returns:
        List of available training dataset function names
    """
    return TrainingDatasetRegistry.list_functions()
