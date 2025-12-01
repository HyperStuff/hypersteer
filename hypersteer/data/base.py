class DatasetFactoryRegistry:
    """Simple registry for dataset factory types"""

    _factories: dict[str, type["BaseDatasetFactory"]] = {}

    @classmethod
    def register(cls, name: str, factory_class: type["BaseDatasetFactory"]):
        """Register a dataset factory class"""
        cls._factories[name] = factory_class

    @classmethod
    def get_factory(cls, name: str) -> type["BaseDatasetFactory"]:
        """Get a registered factory class"""
        if name not in cls._factories:
            raise ValueError(
                f"Unknown dataset factory type: {name}. Available: {list(cls._factories.keys())}"
            )
        return cls._factories[name]

    @classmethod
    def list_factories(cls) -> list:
        """List all registered factory names"""
        return list(cls._factories.keys())


class BaseDatasetFactory:
    """Base class for dataset factories (no abstract methods)"""

    def __init__(self, **kwargs):
        """Initialize the dataset factory with common parameters"""
        pass


def register_factory(name: str):
    """Decorator to register a dataset factory"""

    def decorator(cls):
        DatasetFactoryRegistry.register(name, cls)
        return cls

    return decorator


def get_dataset_factory(factory_type: str, *args, **kwargs):
    """
    Get a dataset factory instance by type.

    Args:
        factory_type: The type of factory to create (e.g., 'axbench')
        *args: Positional arguments to pass to the factory constructor
        **kwargs: Arguments to pass to the factory constructor

    Returns:
        An instance of the requested dataset factory
    """
    factory_class = DatasetFactoryRegistry.get_factory(factory_type)
    return factory_class(*args, **kwargs)


def list_available_factories():
    """
    List all available factory types.

    Returns:
        List of available factory type names
    """
    return DatasetFactoryRegistry.list_factories()
