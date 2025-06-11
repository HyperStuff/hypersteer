from abc import ABC, abstractmethod
from typing import Any

import torch
from pyvene import IntervenableModel
from torch.utils.data import DataLoader

from hypersteer.data.utils import *  # noqa: F403
from hypersteer.utils.configs import ModelConfig, TrainingArgs, WandbConfig
from hypersteer.utils.helpers import get_logger

# Initialize the logger
logger = get_logger(__name__)


class Model(ABC):
    """Abstract base class for all models."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        training_args: TrainingArgs | None = None,
        model_config: ModelConfig | None = None,
        wandb_config: WandbConfig | None = None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = getattr(model_config, "layer", None)
        self.training_args: TrainingArgs = training_args
        self.model_config: ModelConfig = model_config
        self.wandb_config: WandbConfig = wandb_config
        self.max_activations: dict[str, Any] = {}
        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = kwargs.get("seed", 42)
        self.steering_layers = getattr(model_config, "steering_layers", None)
        self.num_of_layers = len(self.steering_layers) if self.steering_layers else 1
        self.dump_dir = kwargs.get("dump_dir", None)

    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of model."""
        pass

    @abstractmethod
    def make_model(self, **kwargs) -> None:
        """Initialize model architecture. Must be implemented by subclasses."""
        pass

    def train(self) -> None:
        """Set the model and any torch modules in this class to train mode."""
        if hasattr(self.model, "train"):
            self.model.train()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.nn.Module) and attr is not self.model:
                attr.train()

    def eval(self) -> None:
        """Set the model and any torch modules in this class to eval mode."""
        if hasattr(self.model, "eval"):
            self.model.eval()
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.nn.Module) and attr is not self.model:
                attr.eval()

    def make_dataloader(self, examples: Any, **kwargs) -> DataLoader:
        """Create a DataLoader for training data."""
        data_module = make_data_module(self.tokenizer, examples, **kwargs)  # noqa: F405
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            data_module["train_dataset"],
            shuffle=True,
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"],
            generator=g,
        )

    @abstractmethod
    def save(self, **kwargs) -> None:
        """Save model state. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Load model state. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict_step(self, batch_examples: Any, batch_idx: int, **kwargs) -> Any:
        """Perform prediction step. Must be implemented by subclasses."""
        raise NotImplementedError

    def to(self, device: str) -> "Model":
        """Move model to specified device."""
        self.device = device
        if hasattr(self, "ax"):
            self.ax = self.ax.to(device)
            if hasattr(self, "ax_model"):
                if isinstance(self.ax_model, IntervenableModel):
                    self.ax_model.set_device(device)
                else:
                    self.ax_model = self.ax_model.to(device)
        return self

    @abstractmethod
    def post_backward(
        self,
        step_outputs: dict[str, Any],
        lr_scheduler: Any,
        optimizer: torch.optim.Optimizer,
        global_step: int,
    ) -> None:
        """Post backward side effects. Must be implemented by subclasses."""
        raise NotImplementedError

    def on_inference_start(self, **kwargs) -> None:
        """Called at the start of inference."""
        pass

    def on_inference_end(self, **kwargs) -> None:
        """Called at the end of inference."""
        pass

    def on_batch_start(self, batch_idx: int, **kwargs) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, batch_idx: int, **kwargs) -> None:
        """Called at the end of each batch."""
        pass
