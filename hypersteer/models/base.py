from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional
import torch


class ModelRegistry:
    """Simple registry for model types"""
    _models: Dict[str, Type['BaseModel']] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type['BaseModel']):
        """Register a model class"""
        cls._models[name] = model_class
    
    @classmethod
    def get_model(cls, name: str) -> Type['BaseModel']:
        """Get a registered model class"""
        if name not in cls._models:
            raise ValueError(f"Unknown model type: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered model names"""
        return list(cls._models.keys())


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def make_model(self, **kwargs):
        pass

    @abstractmethod
    def make_dataloader(self, examples, **kwargs):
        pass

    @abstractmethod
    def train(self, examples, **kwargs):
        pass

    @abstractmethod
    def save(self, dump_dir, **kwargs):
        pass

    @abstractmethod
    def load(self, dump_dir, **kwargs):
        pass

    @abstractmethod
    def predict_steer(self, examples, **kwargs):
        pass

    @abstractmethod
    def get_logits(self, concept_id, k=10):
        pass

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        """Optional method for pre-computing mean activations"""
        pass

    def to(self, device):
        """Optional method for moving model to device"""
        pass


def register_model(name: str):
    """Decorator to register a model"""
    def decorator(cls):
        ModelRegistry.register(name, cls)
        return cls
    return decorator 