from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional
import pandas as pd


class DatasetFactoryRegistry:
    """Simple registry for dataset factory types"""
    _factories: Dict[str, Type['BaseDatasetFactory']] = {}
    
    @classmethod
    def register(cls, name: str, factory_class: Type['BaseDatasetFactory']):
        """Register a dataset factory class"""
        cls._factories[name] = factory_class
    
    @classmethod
    def get_factory(cls, name: str) -> Type['BaseDatasetFactory']:
        """Get a registered factory class"""
        if name not in cls._factories:
            raise ValueError(f"Unknown dataset factory type: {name}. Available: {list(cls._factories.keys())}")
        return cls._factories[name]
    
    @classmethod
    def list_factories(cls) -> list:
        """List all registered factory names"""
        return list(cls._factories.keys())


class BaseDatasetFactory(ABC):
    """Abstract base class for dataset factories"""
    
    def __init__(self, **kwargs):
        """Initialize the dataset factory with common parameters"""
        pass
    
    @abstractmethod
    def save_cache(self):
        """Save the language model cache before exiting"""
        pass
    
    @abstractmethod
    def reset_stats(self):
        """Reset API costs"""
        pass
    
    @abstractmethod
    def prepare_genre_concepts(self, concepts, **kwargs):
        """Prepare genre concepts for the given concepts"""
        pass
    
    @abstractmethod
    def prepare_concepts(self, concepts, **kwargs):
        """Prepare concepts and contrast concepts"""
        pass
    
    @abstractmethod
    def create_eval_df(self, concepts, subset_n, concept_genres_map, 
                      train_contrast_concepts_map, eval_contrast_concepts_map, 
                      mode="balance", **kwargs) -> pd.DataFrame:
        """Create evaluation dataframe"""
        pass
    
    @abstractmethod
    def create_train_df(self, concept, n, concept_genres_map, **kwargs) -> pd.DataFrame:
        """Create training dataframe"""
        pass
    
    @abstractmethod
    def create_dpo_df(self, existing_df, **kwargs) -> pd.DataFrame:
        """Create DPO dataframe"""
        pass
    
    def create_imbalance_eval_df(self, subset_n, factor=100) -> pd.DataFrame:
        """Create imbalanced evaluation dataframe (optional implementation)"""
        raise NotImplementedError("create_imbalance_eval_df not implemented for this factory")


class BaseSteeringDatasetFactory(ABC):
    """Abstract base class for steering dataset factories"""
    
    def __init__(self, **kwargs):
        """Initialize the steering dataset factory"""
        pass
    
    @abstractmethod
    def create_eval_df(self, concepts, subset_n, steering_factors, 
                      steering_datasets, concept_id, steering_model_name, **kwargs) -> pd.DataFrame:
        """Create evaluation dataframe for steering"""
        pass
    
    def augment_train_df_with_steered_prompts(self, train_df, concepts, **kwargs) -> pd.DataFrame:
        """Augment training dataframe with steered prompts (optional implementation)"""
        raise NotImplementedError("augment_train_df_with_steered_prompts not implemented for this factory")


def register_factory(name: str):
    """Decorator to register a dataset factory"""
    def decorator(cls):
        DatasetFactoryRegistry.register(name, cls)
        return cls
    return decorator 