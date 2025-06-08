import gc
import itertools
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_scheduler

import wandb
from hypersteer.utils.configs import ModelConfig, TrainingArgs, WandbConfig
from hypersteer.utils.helpers import get_logger, get_rank, get_world_size

logger = get_logger(__name__)


class TrainerMixin(ABC):
    """Abstract mixin that defines the interface for trainable models."""

    @abstractmethod
    def setup_model(self, **kwargs) -> None:
        """Setup the model for training."""
        pass

    @abstractmethod
    def setup_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        """Setup and return the optimizer."""
        pass

    @abstractmethod
    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a single training step.

        Args:
            batch: The input batch

        Returns:
            Dictionary containing loss and any other metrics
        """
        pass

    @abstractmethod
    def val_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Perform a single validation step.

        Args:
            batch: The input batch

        Returns:
            Dictionary containing loss and any other metrics
        """
        pass

    def log_metrics(self, metrics, mode="train"):
        """Log metrics to wandb and logger."""
        # Prepare log dictionary
        log_dict = {}

        for key_tuple, value in metrics.items():
            section, keyname = key_tuple
            # Determine the log key format based on section
            # Special case for 'counters' section (e.g., step, lr)
            if section == "counters":
                log_key = f"{section}/{keyname}"
            # General case: {section}_{mode}/keyname
            else:
                log_key = f"{section}_{mode}/{keyname}"

            # Handle value extraction (for tensors, etc.)
            if isinstance(value, torch.Tensor):
                processed_value = value.detach().item()
            else:
                processed_value = float(value)  # Ensure it's a standard float

            log_dict[log_key] = processed_value

        # Log to wandb
        if wandb.run and (not dist.is_initialized() or dist.get_rank() == 0):
            wandb.log(log_dict)

        logger.info(log_dict)

    @abstractmethod
    def get_trainable_parameters(self):
        """Return the trainable parameters for gradient clipping."""
        pass

    def on_train_epoch_start(self, epoch: int) -> None:
        """Called at the start of each training epoch."""
        pass

    def on_train_epoch_end(self, epoch: int) -> None:
        """Called at the end of each training epoch."""
        pass

    def on_validation_start(self, global_step: int) -> None:
        """Called at the start of validation."""
        pass

    def on_validation_end(self, global_step: int) -> None:
        """Called at the end of validation."""
        pass


class Trainer:
    """Generic trainer that works with any model implementing TrainerMixin."""

    def __init__(
        self: "Trainer",
        model: TrainerMixin,
        training_args: TrainingArgs,
        model_config: ModelConfig,
        wandb_config: WandbConfig | None = None,
        device=None,
        seed=42,
    ) -> None:
        self.model = model
        self.training_args = training_args
        self.model_config = model_config
        self.wandb_config = wandb_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = seed

        # Training state
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.global_val_step = 0

        # Distributed training
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.is_distributed = self.world_size > 1

    def setup_training(
        self: "Trainer",
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader | None = None,
    ) -> tuple[int, int]:
        """Setup training components."""
        # Setup model
        self.model.setup_model()

        # Setup optimizer
        self.optimizer = self.model.setup_optimizer()

        # Determine training configuration
        use_step_limit = self.training_args.n_steps > 0
        if use_step_limit:
            num_training_steps = self.training_args.n_steps
            effective_epochs = float("inf")
            logger.info(f"Training with step limit: {num_training_steps} steps")
        else:
            num_training_steps = self.training_args.n_epochs * (
                len(train_dataloader) // self.training_args.gradient_accumulation_steps
            )
            effective_epochs = self.training_args.n_epochs
            logger.info(f"Training with epoch limit: {effective_epochs} epochs")

        # Setup learning rate scheduler
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.training_args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Log training configuration
        steps_per_epoch = (
            len(train_dataloader) // self.training_args.gradient_accumulation_steps
        )
        logger.info("Training configuration:")
        logger.info(f"  - Batch size: {self.training_args.batch_size}")
        logger.info(
            f"  - Gradient accumulation steps: {self.training_args.gradient_accumulation_steps}"
        )
        logger.info(f"  - Steps per epoch: {steps_per_epoch}")
        logger.info(f"  - Total training steps: {num_training_steps}")
        if use_step_limit:
            estimated_epochs = num_training_steps / steps_per_epoch
            logger.info(f"  - Estimated epochs to complete: {estimated_epochs:.2f}")
        else:
            logger.info(f"  - Training epochs: {effective_epochs}")

        # Setup wandb watching
        if wandb.run and self.wandb_config and self.wandb_config.watch_grads:
            # Let the model decide what to watch
            if hasattr(self.model, "get_watchable_modules"):
                modules = self.model.get_watchable_modules()
                wandb.watch(modules, log_freq=self.wandb_config.watch_grads_freq)

        return num_training_steps, effective_epochs

    def train(
        self,
        train_dataloader: DataLoader,
        dev_dataloader: DataLoader | None = None,
        train_sampler: DistributedSampler | None = None,
    ):
        """Main training loop."""
        num_training_steps, effective_epochs = self.setup_training(
            train_dataloader, dev_dataloader
        )
        use_step_limit = self.training_args.n_steps > 0

        # Training state
        accum_counter = 0
        epoch = 0

        # Training loop
        while epoch < effective_epochs:
            self.model.on_train_epoch_start(epoch)

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_iter = itertools.cycle(train_dataloader)
            step = 0

            while step < len(train_dataloader):
                # Check if we've reached the step limit
                if use_step_limit and self.global_step >= num_training_steps:
                    logger.info(
                        f"Reached step limit of {num_training_steps} steps. Terminating training."
                    )
                    return

                # Get next batch
                batch = next(train_iter)

                # Training step
                if accum_counter == 0:
                    self.optimizer.zero_grad()

                # Forward pass
                step_outputs = self.model.train_step(batch, self.global_step)
                loss = step_outputs[("loss", "main")]

                # Backward pass
                scaled_loss = loss / self.training_args.gradient_accumulation_steps
                scaled_loss.backward()
                accum_counter += 1

                # Optimizer step
                if accum_counter == self.training_args.gradient_accumulation_steps:
                    step_outputs = self.model.post_backward(
                        step_outputs,
                        self.lr_scheduler,
                        self.optimizer,
                        self.global_step,
                    )
                    accum_counter = 0

                    self.optimizer.step()
                    self.lr_scheduler.step()

                    # Log metrics
                    step_outputs[("counters", "global_step")] = self.global_step
                    self.model.log_metrics(step_outputs, mode="train")

                step += 1
                self.global_step += 1

                # Validation
                if (
                    dev_dataloader is not None
                    and hasattr(self.training_args, "val_interval")
                    and self.training_args.val_interval > 0
                    and self.global_step % self.training_args.val_interval == 0
                ):
                    self.validate(dev_dataloader)

                # Cleanup
                del batch, step_outputs, loss, scaled_loss
                torch.cuda.empty_cache()

            self.model.on_train_epoch_end(epoch)
            epoch += 1

    @torch.no_grad()
    def validate(self, dev_dataloader: DataLoader):
        """Run validation."""
        logger.info(f"Running validation at step {self.global_step}")

        self.model.on_validation_start()

        # Set to eval mode
        if hasattr(self.model, "eval"):
            self.model.eval()

        all_metrics = []

        for batch in dev_dataloader:
            step_outputs = self.model.val_step(batch, self.global_step)
            all_metrics.append(step_outputs)

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        aggregated_metrics[("counters", "val_step")] = self.global_step
        aggregated_metrics[("counters", "global_val_step")] = self.global_val_step

        # Log validation metrics
        self.model.log_metrics(aggregated_metrics, mode="val")

        self.global_val_step += 1

        # Set back to train mode
        if hasattr(self.model, "train"):
            self.model.train()

        self.model.on_validation_end()

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    def _aggregate_metrics(self, metrics_list):
        """Aggregate metrics from multiple validation steps."""
        if not metrics_list:
            return {}

        aggregated = {}
        for _, key in metrics_list[0].keys():
            if key in ["loss", "grad_norm"] or key.endswith("_loss"):
                # Average numerical metrics
                values = [
                    m[key] for m in metrics_list if key in m and m[key] is not None
                ]
                if values:
                    aggregated[key] = sum(values) / len(values)
            elif key.endswith("_count"):
                # Sum count metrics
                values = [
                    m[key] for m in metrics_list if key in m and m[key] is not None
                ]
                if values:
                    aggregated[key] = sum(values)

        return aggregated
