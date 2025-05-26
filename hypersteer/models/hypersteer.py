import gc
import json
import os
import random

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import wandb
from pyvene import IntervenableConfig, IntervenableModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from hypersteer.data.utils import get_batch_locs, make_data_module
from hypersteer.training import TrainerMixin
from hypersteer.utils.helpers import (
    configure_tokenizer_model,
    get_logger,
    get_rank,
    load_metadata,
)
from hypersteer.utils.patch import monkeypatch_ax_model_generate
from hypersteer.utils.visualization import Visualizer

from .base import register_model
from .hypernet.configuration_hypernet import HypernetConfig
from .hypernet.modeling_hypernet import HypernetModel
from .interventions import HyperAdditiveIntervention
from .model import Model

logger = get_logger(__name__)


def load_concept_id_to_desecription(metadata_path):
    concept_id_to_description = {}
    with open(metadata_path) as f:
        for line in f:
            data = json.loads(line)
            concept_id, description = data["concept_id"], data["concept"]
            concept_id_to_description[concept_id] = description
    return concept_id_to_description


def partition_df(df, n):
    total_rows = len(df)
    partition_size = total_rows // n
    remainder = total_rows % n
    partitions = []
    start_idx = 0
    for i in range(n):
        current_size = partition_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_size
        partitions.append(df.iloc[start_idx:end_idx].copy())
        start_idx = end_idx
    return partitions


class RegressionWrapper(nn.Module):
    def __init__(self, base_model, hidden_size, output_dim):
        super().__init__()
        self.base_model = base_model
        self.regression_head = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        input_ids,
        attention_mask,
        output_attentions=False,
        normalize=False,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=True,
        )
        if isinstance(outputs, tuple):
            last_hiddens = outputs[0].hidden_states[-1]
        else:
            last_hiddens = outputs.hidden_states[-1]
        last_token_representations = last_hiddens[:, -1]
        preds = self.regression_head(last_token_representations)
        if normalize:
            preds = F.normalize(preds, p=2, dim=-1)
        if output_attentions:
            return preds, outputs[1:][1]
        return preds


@register_model("HyperSteer")
class HyperSteer(Model, TrainerMixin):
    """Base HyperSteer model implementation. Supports various hypernet types."""

    def __str__(self):
        return "HyperSteer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visualizer = Visualizer()

    def make_model(self, **kwargs):
        self.ax = HyperAdditiveIntervention(
            embed_dim=self.model.config.hidden_size,
            low_rank_dimension=self.model_config.low_rank_dimension,
            use_selection_head=self.model_config.use_selection_head,
            use_ln=self.model_config.use_selection_ln,
        ).to(self.device)

        self.ax.train()

        layers = self.steering_layers if self.steering_layers else [self.layer]
        if layers == "all":
            layers = list(range(self.model.config.num_hidden_layers))
        self.num_of_layers = len(layers)
        ax_config = IntervenableConfig(
            representations=[
                {
                    "layer": layer,
                    "component": f"model.layers[{layer}].output",
                    "low_rank_dimension": self.model_config.low_rank_dimension,
                    "intervention": self.ax,
                }
                for layer in layers
            ]
        )
        ax_model = IntervenableModel(ax_config, self.model)
        ax_model.set_device(self.device)
        self.ax_model = ax_model

        self.base_model_tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_name, model_max_length=512
        )
        self.base_model_tokenizer.padding_side = "left"
        base_model_config = AutoConfig.from_pretrained(
            self.model_config.base_model_name
        )
        if self.model_config.hypernet_type == "regression":
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model_name, torch_dtype=torch.bfloat16
            )
            configure_tokenizer_model(base_model, self.base_model_tokenizer)
            self.concept_embedding = RegressionWrapper(
                base_model=base_model,
                hidden_size=base_model.config.hidden_size,
                output_dim=self.model.config.hidden_size,
            )
        elif self.model_config.hypernet_type == "attn":
            hypernet_config = HypernetConfig(
                num_hidden_layers=self.model_config.cross_attn_hidden_layers,
                target_model_name_or_path=self.model_config.base_model_name,
                hidden_size=base_model_config.hidden_size,
                torch_dtype=torch.bfloat16,
            )
            self.concept_embedding = HypernetModel(config=hypernet_config)

        self.concept_embedding = self.concept_embedding.to(
            self.device, dtype=torch.bfloat16
        )
        if self.model_config.do_reconstruction:
            metadata_path = os.path.join(
                self.model_config.reconstruction_dict_path, "metadata.jsonl"
            )
            self.concept_id_to_text = load_concept_id_to_desecription(metadata_path)
        meta_data = kwargs.get("metadata", None)
        if meta_data is not None:
            self.concept_id_to_text = {}
            for d in meta_data:
                self.concept_id_to_text[d["concept_id"]] = d["concept"]
        self.eval_steps = kwargs.get("eval_steps", 20)
        self.log_per_step = kwargs.get("log_per_step", 10)
        self.include_sentence_in_embedding = kwargs.get(
            "include_sentence_in_embedding", False
        )
        self.metadata = load_metadata(kwargs.get("metadata_path"))

    # TrainerMixin implementation
    def setup_model(self, training_args, model_config, **kwargs):
        """Setup the model for training."""
        self.concept_embedding = self._setup_model_for_distributed(
            self.concept_embedding
        )
        self.concept_embedding.train()
        self.ax.train()

    def setup_optimizer(self, training_args):
        """Setup and return the optimizer."""
        optimizer = torch.optim.AdamW(
            self.concept_embedding.parameters(),
            lr=training_args.lr,
            weight_decay=training_args.weight_decay,
        )
        return optimizer

    def get_trainable_parameters(self):
        """Return parameters that should be included in gradient clipping."""
        return self.concept_embedding.parameters()

    def get_watchable_modules(self):
        """Return modules that should be watched by wandb."""
        return [self.ax, self.concept_embedding]

    def train_step(self, batch, global_step):
        """Perform a single training step."""
        inputs = {k: v.to(self.device) for k, v in batch.items()}

        unit_locations = {
            "sources->base": (
                None,
                inputs["intervention_locations"]
                .permute(1, 0, 2)
                .expand(self.num_of_layers, -1, -1)
                .tolist(),
            )
        }
        subspaces = [{"k": self.model_config.topk} for _ in range(self.num_of_layers)]

        # Compute main loss
        loss = self.compute_main_loss_and_outputs(inputs, unit_locations, subspaces)

        step_outputs = {"loss": loss}

        # Add selection sparsity loss if enabled
        if self.model_config.use_selection_head:
            mask = self.ax_model.full_intervention_outputs[0].payload["mask"]
            selection_sparsity_loss = self.compute_selection_sparsity_loss(
                gathered_sparse_mask=mask,
                attention_mask=inputs["attention_mask"],
                locs=inputs["intervention_locations"],
            )
            loss += selection_sparsity_loss * self.training_args.selection_l1_loss_coeff
            step_outputs["selection_sparsity_loss"] = selection_sparsity_loss
            step_outputs["loss"] = loss  # Update with total loss

            # Visualize sparse mask at the configured frequency
            self._visualize_training_mask(mask, inputs, global_step)

        return step_outputs

    def val_step(self, batch, global_step):
        """Perform a single validation step."""
        inputs = {k: v.to(self.device) for k, v in batch.items()}

        unit_locations = {
            "sources->base": (
                None,
                inputs["intervention_locations"]
                .permute(1, 0, 2)
                .expand(self.num_of_layers, -1, -1)
                .tolist(),
            )
        }
        subspaces = [{"k": self.model_config.topk} for _ in range(self.num_of_layers)]
        if self.model_config.use_selection_head:
            for subspace in subspaces:
                subspace.update({"locs": inputs["intervention_locations"]})

        # Compute concept embedding
        if self.model_config.hypernet_type == "regression":
            v = self.concept_embedding(
                inputs["concept_input_ids"],
                inputs["concept_attention_mask"],
            )
        elif self.model_config.hypernet_type == "attn":
            concept_inputs_embeds = self.model.model.embed_tokens(
                inputs["concept_input_ids"]
            )
            base_intervention_mask = inputs["labels"] == -100
            base_intervention_mask = base_intervention_mask & inputs["attention_mask"]
            base_hidden_state = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            ).hidden_states[self.layer]
            v = self.concept_embedding(
                input_ids=None,
                inputs_embeds=concept_inputs_embeds,
                attention_mask=inputs["concept_attention_mask"],
                base_encoder_hidden_states=base_hidden_state,
                base_encoder_attention_mask=base_intervention_mask,
                output_hidden_states=False,
            ).last_hidden_state

        self.ax._update_v(v)
        base_out, cf_out = self.ax_model(
            base={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=subspaces,
            use_cache=False,
            output_original_output=True,
        )

        steering_loss = cf_out.loss
        step_outputs = {"loss": steering_loss}

        # Handle selection head validation
        if self.model_config.use_selection_head:
            gathered_sparse_mask = self.ax_model.full_intervention_outputs[0].payload[
                "mask"
            ]
            selection_sparsity_loss = self.compute_selection_sparsity_loss(
                gathered_sparse_mask=gathered_sparse_mask,
                attention_mask=inputs["attention_mask"],
                locs=inputs["intervention_locations"],
            )
            step_outputs["selection_sparsity_loss"] = selection_sparsity_loss

            # Visualize validation mask
            self._visualize_validation_mask(gathered_sparse_mask, inputs, global_step)

        # Logit diff visualization
        if (
            self.model_config.logit_diff_visualization.log_heatmap
            and global_step
            % self.model_config.logit_diff_visualization.log_heatmap_freq
            == 0
        ):
            self._logit_diff_visualization(
                base_out,
                cf_out,
                inputs,
                mode="val/logit_diff",
                step=global_step,
                normalize=False,
                dump_dir=self.dump_dir,
            )

        self.ax._reset_v()
        return step_outputs

    def log_metrics(self, metrics, mode="train"):
        """Log metrics to wandb and logger."""
        # Prepare log dictionary
        log_dict = {}

        for key, value in metrics.items():
            if key == "global_step":
                log_dict["counters/step"] = value
            elif key == "lr":
                log_dict["counters/lr"] = value
            elif key == "grad_norm":
                log_dict[f"{mode}/main_grad_norm"] = (
                    value.item() if hasattr(value, "item") else float(value)
                )
            elif key == "loss":
                log_dict[f"{mode}/loss"] = (
                    value.detach().item() if hasattr(value, "detach") else float(value)
                )
            elif key.endswith("_loss"):
                log_dict[f"{mode}/{key}"] = (
                    value.detach().item() if hasattr(value, "detach") else float(value)
                )
            elif key in ["val_step", "global_val_step"]:
                log_dict[key] = value
            else:
                log_dict[f"{mode}/{key}"] = value

        # Log to wandb
        if wandb.run and (not dist.is_initialized() or dist.get_rank() == 0):
            wandb.log(log_dict)

        logger.info(log_dict)

    def on_validation_start(self, global_step):
        """Called at the start of validation."""
        self.concept_embedding.eval()
        self.ax.eval()

    def on_validation_end(self, global_step):
        """Called at the end of validation."""
        self.concept_embedding.train()
        self.ax.train()

    # Helper methods for visualization
    def _visualize_training_mask(self, mask, inputs, global_step):
        """Visualize training mask."""
        batch_tokens = [
            self.tokenizer.convert_ids_to_tokens(input_id)
            for input_id in inputs["input_ids"]
        ]
        concept_strings = []
        for concept_id in inputs["concept_ids"]:
            concept_str = f"concept_{concept_id.item()}"
            if hasattr(self, "metadata") and self.metadata is not None:
                for meta_item in self.metadata:
                    if meta_item.get("concept_id") == concept_id.item():
                        concept_str = meta_item.get("concept", concept_str)
                        break
            concept_strings.append(concept_str)

        self._visualize_token_heatmap(
            mask.squeeze(),
            step=global_step,
            dump_dir=self.dump_dir or "assets/cache/sparse_masks",
            batch_tokens=batch_tokens,
            concept_ids=inputs["concept_ids"].tolist(),
            concept_strings=concept_strings,
            attention_mask=inputs["attention_mask"],
            viz_mode="train/mask",
            title_prefix="Training",
        )

    def _visualize_validation_mask(self, mask, inputs, global_step):
        """Visualize validation mask."""
        batch_tokens = [
            self.tokenizer.convert_ids_to_tokens(input_id)
            for input_id in inputs["input_ids"]
        ]
        concept_strings = []
        for concept_id in inputs["concept_ids"]:
            concept_str = f"concept_{concept_id.item()}"
            if hasattr(self, "metadata") and self.metadata is not None:
                for meta_item in self.metadata:
                    if meta_item.get("concept_id") == concept_id.item():
                        concept_str = meta_item.get("concept", concept_str)
                        break
            concept_strings.append(concept_str)

        self._visualize_token_heatmap(
            mask.squeeze(),
            step=global_step,
            dump_dir=self.dump_dir or "assets/cache/sparse_masks",
            batch_tokens=batch_tokens,
            concept_ids=inputs["concept_ids"].tolist(),
            concept_strings=concept_strings,
            attention_mask=inputs["attention_mask"],
            viz_mode="val/mask",
            title_prefix="Validation",
        )

    def _setup_model_for_distributed(self, model):
        """Setup model for distributed training if needed."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = get_rank()
            device = torch.device(f"cuda:{rank}")
            model = model.to(device)
            model = DDP(model, device_ids=[rank])
        return model

    def compute_main_loss_and_outputs(
        self,
        inputs,
        unit_locations,
        subspaces,
    ):
        if self.model_config.hypernet_type == "regression":
            v = self.concept_embedding(
                inputs["concept_input_ids"],
                inputs["concept_attention_mask"],
            )
        elif self.model_config.hypernet_type == "attn":
            concept_inputs_embeds = self.model.model.embed_tokens(
                inputs["concept_input_ids"]
            )
            base_intervention_mask = inputs["labels"] == -100
            base_intervention_mask = base_intervention_mask & inputs["attention_mask"]
            base_hidden_state = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            ).hidden_states[self.layer]
            v = self.concept_embedding(
                input_ids=None,
                inputs_embeds=concept_inputs_embeds,
                attention_mask=inputs["concept_attention_mask"],
                base_encoder_hidden_states=base_hidden_state,
                base_encoder_attention_mask=base_intervention_mask,
                output_hidden_states=False,
            ).last_hidden_state

        self.ax._update_v(v)
        base_output, cf_outputs = self.ax_model(
            base={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
            unit_locations=unit_locations,
            labels=inputs["labels"],
            subspaces=subspaces,
            use_cache=False,
            output_original_output=True,
        )
        self.ax._reset_v()

        steering_loss = cf_outputs.loss
        loss = steering_loss
        return loss

    def compute_selection_sparsity_loss(
        self,
        gathered_sparse_mask=None,
        attention_mask=None,
        locs=None,
    ):
        # Scatter sparse mask to right locs in padded window
        if locs is not None:
            sparse_mask = torch.zeros_like(
                attention_mask,
                dtype=gathered_sparse_mask.dtype,
                device=gathered_sparse_mask.device,
            )

            sparse_mask.scatter_(
                -1, locs.squeeze(1).long(), gathered_sparse_mask.squeeze(-1)
            )

        total_counts = attention_mask.sum(-1)
        # L1 loss
        aux_loss = (sparse_mask * attention_mask).abs().sum(
            -1
        ) / total_counts.clamp_min(1)

        return aux_loss.mean()

    def _visualize_token_heatmap(
        self,
        mask,
        step=None,
        dump_dir=None,
        batch_tokens=None,
        concept_ids=None,
        concept_strings=None,
        attention_mask=None,
        viz_mode="sparse_mask",
        title_prefix=None,
    ):
        """
        Helper to visualize sparse mask using the configured visualization options.
        """
        if not self.model_config.mask_visualization.log_heatmap:
            return
        freq = self.model_config.mask_visualization.log_heatmap_freq
        # If step is a tuple (step, freq), use step[0] for step and step[1] for freq
        if isinstance(step, tuple):
            step, freq = step
        if step is not None and freq is not None and step % freq != 0:
            return

        self.visualizer.log_visualization(
            mask,
            step=step,
            dump_dir=dump_dir or self.dump_dir or "assets/cache/sparse_masks",
            batch_tokens=batch_tokens,
            concept_ids=concept_ids,
            concept_strings=concept_strings,
            attention_mask=attention_mask,
            viz_mode=viz_mode,
            pdf_visualization=self.model_config.mask_visualization.pdf_visualization,
            png_visualization=self.model_config.mask_visualization.png_visualization,
            title_prefix=title_prefix or "Sparse Mask",
            log_all_examples=self.model_config.mask_visualization.log_all_examples,
            log_to_console=False,
        )

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        raise NotImplementedError(
            "predict_latent is not implemented. Latent logic has been removed."
        )

    def make_dataloader(
        self, examples, rank, world_size, shuffle=True, distributed=False, **kwargs
    ):
        if distributed:
            sampler = DistributedSampler(
                examples, num_replicas=world_size, rank=rank, shuffle=shuffle
            )
            data_module = make_data_module(self.tokenizer, examples, **kwargs)
            g = torch.Generator()
            g.manual_seed(self.seed)
            train_dataloader = DataLoader(
                data_module["train_dataset"],
                batch_size=self.training_args.batch_size,
                collate_fn=data_module["data_collator"],
                sampler=sampler,
                generator=g,
                drop_last=False,
            )
        else:
            data_module = make_data_module(self.tokenizer, examples, **kwargs)
            g = torch.Generator()
            g.manual_seed(self.seed)
            train_dataloader = DataLoader(
                data_module["train_dataset"],
                batch_size=self.training_args.batch_size,
                collate_fn=data_module["data_collator"],
                shuffle=shuffle,
                generator=g,
                drop_last=False,
            )
            sampler = None
        return train_dataloader, sampler

    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = os.path.join(dump_dir, f"{model_name}_weight.pt")
        weight = self.concept_embedding.cpu()
        torch.save(weight, weight_file)

        # Save token selection (sparse_selection) if enabled
        if hasattr(self.ax, "selection_head") and self.model_config.use_selection_head:
            path = os.path.join(dump_dir, f"{model_name}_selection_head.pt")
            torch.save(self.ax.selection_head.state_dict(), path)
            logger.debug(f"Saved selection head to {path}")

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = os.path.join(dump_dir, f"{model_name}_weight.pt")
        self.make_model(**kwargs)

        del self.concept_embedding
        self.concept_embedding = torch.load(
            weight_file, map_location=self.device, weights_only=False
        )

        # Load token selection (sparse_selection) if enabled and file exists
        if self.model_config.use_selection_head and hasattr(self.ax, "selection_head"):
            path = os.path.join(dump_dir, f"{model_name}_selection_head.pt")
            if os.path.exists(path):
                self.ax.selection_head.load_state_dict(
                    torch.load(path, map_location=self.device)
                )
                logger.debug(f"Loaded selection head from {path}")

    def get_logits(self, concept_id, metadata=None, k=10):
        top_logits, neg_logits = [None], [None]

        W_U = self.model.lm_head.weight.T
        W_U = (
            W_U
            * (
                self.model.model.norm.weight
                + torch.ones_like(self.model.model.norm.weight)
            )[:, None]
        )
        W_U -= einops.reduce(W_U, "d_model d_vocab -> 1 d_vocab", "mean")

        if metadata is not None:
            concept_text = None
            for d in metadata:
                if d["concept_id"] == concept_id:
                    concept_text = d["concept"]
                    break
            if concept_text is None:
                raise ValueError("Concept ID not found in metadata.")

        else:
            concept_text = self.concept_id_to_text[concept_id]

        concept_input = self.base_model_tokenizer(
            concept_text,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
        ).to(self.device)

        concept_subspace = self.concept_embedding(
            concept_input["input_ids"],
            concept_input["attention_mask"],
        )

        vocab_logits = concept_subspace @ W_U
        top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
        top_tokens = self.tokenizer.batch_decode(top_indices)

        top_logits = [list(zip(top_tokens, top_values.tolist()))]

        neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
        neg_tokens = self.tokenizer.batch_decode(neg_indices)
        neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]

        return top_logits, neg_logits

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax_model = monkeypatch_ax_model_generate(self.ax_model)
        self.ax.eval()
        self.tokenizer.padding_side = "left"
        self.dump_dir = kwargs.get("dump_dir", None)
        concept_id_col = (
            "sae_id"
            if "sae" in self.__str__().lower()
            and not kwargs.get("disable_neuronpedia_max_act", False)
            else "concept_id"
        )
        use_synergy = kwargs.get("use_synergy", False)
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        all_generations = []
        all_perplexities = []
        all_strengths = []
        all_steering_vectors = []  # <-- Add this line
        total_batches = (len(examples) + batch_size - 1) // batch_size
        infer_dump_dir = os.path.join(
            kwargs.get("dump_dir") or "assets/cache/sparse_masks", "inference_steer"
        )
        os.makedirs(infer_dump_dir, exist_ok=True)

        cross_attn_dump_dir = os.path.join(
            kwargs.get("dump_dir") or "assets/cache/sparse_masks", "cross_attn_heatmaps"
        )
        os.makedirs(cross_attn_dump_dir, exist_ok=True)

        with torch.inference_mode():
            for batch_idx, i in enumerate(
                tqdm(
                    range(0, len(examples), batch_size),
                    desc="Generating steered text",
                    total=total_batches,
                )
            ):
                batch_examples = examples.iloc[i : i + batch_size]
                if use_synergy:
                    input_strings = batch_examples["steered_input"].tolist()
                else:
                    input_strings = batch_examples["input"].tolist()
                mag = torch.tensor(batch_examples["factor"].tolist()).to(self.device)
                idx = torch.tensor(batch_examples["concept_id"].tolist()).to(
                    self.device
                )
                max_acts = torch.tensor(
                    [
                        self.max_activations.get(id, 1.0)
                        for id in batch_examples[concept_id_col].tolist()
                    ]
                ).to(self.device)
                inputs = self.tokenizer(
                    input_strings, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

                if self.model_config.include_sentence_in_embedding:
                    input_concept = []
                    concepts = batch_examples["input_concept"].tolist()
                    inputs_list = batch_examples["input"].tolist()
                    assert len(concepts) == len(inputs_list)
                    for concept, input_string in zip(concepts, inputs_list):
                        input_concept.append(concept + " [Input] " + input_string)
                else:
                    input_concept = batch_examples["input_concept"].tolist()

                concept_inputs = self.base_model_tokenizer(
                    input_concept,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                ).to(self.device)

                # --- Concept embedding (v) ---
                if self.model_config.hypernet_type == "regression":
                    v = self.concept_embedding(
                        concept_inputs["input_ids"],
                        concept_inputs["attention_mask"],
                    )
                elif self.model_config.hypernet_type == "attn":
                    concept_inputs_embeds = self.model.model.embed_tokens(
                        concept_inputs["input_ids"]
                    )
                    base_intervention_mask = inputs["attention_mask"]
                    base_hidden_state = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True,
                    ).hidden_states[self.layer]

                    if self.model_config.cross_attn_heatmap_visualization.log_heatmap:
                        outputs = self.concept_embedding(
                            input_ids=None,
                            inputs_embeds=concept_inputs_embeds,
                            attention_mask=concept_inputs["attention_mask"],
                            base_encoder_hidden_states=base_hidden_state,
                            base_encoder_attention_mask=base_intervention_mask,
                            output_hidden_states=False,
                            output_attentions=True,
                            return_dict=True,
                        )
                        v = outputs.last_hidden_state
                        cross_attn_weights = outputs.cross_attentions
                        freq = self.model_config.cross_attn_heatmap_visualization.log_heatmap_freq
                        if batch_idx % freq == 0:
                            self._save_cross_attn_heatmaps(
                                cross_attn_weights,
                                inputs["input_ids"],
                                concept_inputs["input_ids"],
                                inputs["attention_mask"],  # concept_attention_mask
                                concept_inputs[
                                    "attention_mask"
                                ],  # input_attention_mask
                                cross_attn_dump_dir,
                                batch_idx,
                                tokenizer=self.tokenizer,
                                prefix="cross_attn",
                            )
                    else:
                        v = self.concept_embedding(
                            input_ids=None,
                            inputs_embeds=concept_inputs_embeds,
                            attention_mask=concept_inputs["attention_mask"],
                            base_encoder_hidden_states=base_hidden_state,
                            base_encoder_attention_mask=base_intervention_mask,
                            output_hidden_states=False,
                        ).last_hidden_state

                # Store steering vectors for each example in the batch (move to cpu)
                v_np = v.detach().float().cpu().numpy()
                for row in v_np:
                    all_steering_vectors.append(row.copy())

                self.ax._update_v(v)

                locs = get_batch_locs(
                    prefix_length=kwargs["prefix_length"],
                    inputs=batch_examples["input"],
                    attention_mask=inputs["attention_mask"],
                    tokenizer=self.tokenizer,
                )

                # Always define subspaces as a list of dicts (one per layer)
                subspaces = [
                    {
                        "idx": idx,
                        "mag": mag,
                        "max_act": max_acts,
                        "prefix_length": kwargs["prefix_length"],
                        "locs": locs.to(self.device),
                    }
                    for _ in range(self.num_of_layers)
                ]

                # Generate with intervention (steered)
                base_out, steered_out = self.ax_model.generate(
                    inputs,
                    unit_locations=None,
                    intervene_on_prompt=True,
                    subspaces=subspaces,
                    max_new_tokens=eval_output_length,
                    do_sample=True,
                    temperature=temperature,
                    output_original_output=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Get generated sequences
                if isinstance(steered_out, dict):
                    generations = steered_out.get("sequences", None)
                else:
                    generations = getattr(steered_out, "sequences", None)

                # Forward pass through ax_model to get full logits for generated sequences
                gen_attention_mask = (generations != self.tokenizer.pad_token_id).long()
                with torch.no_grad():
                    base_out_gen, steered_out_gen = self.ax_model(
                        base={
                            "input_ids": generations,
                            "attention_mask": gen_attention_mask,
                        },
                        unit_locations=None,
                        labels=None,
                        subspaces=subspaces,
                        use_cache=False,
                        output_original_output=True,
                    )
                # Logit diff visualization every N batches (on generated text)
                if (
                    self.model_config.logit_diff_visualization.log_heatmap
                    and batch_idx
                    % self.model_config.logit_diff_visualization.log_heatmap_freq
                    == 0
                ):
                    self._logit_diff_visualization(
                        base_out_gen,
                        steered_out_gen,
                        {
                            "input_ids": generations,
                            "attention_mask": gen_attention_mask,
                        },
                        mode="steer/logit_diff",
                        step=batch_idx,
                        normalize=False,
                        dump_dir=self.dump_dir,
                    )

                del base_out, steered_out, base_out_gen, steered_out_gen
                gc.collect()
                torch.cuda.empty_cache()

                # Construct the flat viz-ready mask for generated text
                if self.model_config.use_selection_head:
                    gathered_sparse_mask = self.ax_model.full_intervention_outputs[
                        0
                    ].payload.get("mask", None)

                    if (
                        gathered_sparse_mask is not None
                        and self.training_args.use_selection_head
                    ):
                        for output in self.ax_model.full_intervention_outputs[1:]:
                            current_mask = output.payload.get("mask", None)
                            if (
                                current_mask is not None
                                and gathered_sparse_mask.shape[0]
                                == current_mask.shape[0]
                            ):
                                gathered_sparse_mask = torch.cat(
                                    [gathered_sparse_mask, current_mask], dim=-1
                                )
                    # Visualize mask if enabled
                    if (
                        gathered_sparse_mask is not None
                        and self.model_config.use_selection_head
                        and self.model_config.use_selection_head
                    ):
                        batch_tokens = []
                        for generated_ids in generations:
                            valid_tokens = self.tokenizer.convert_ids_to_tokens(
                                generated_ids
                            )
                            batch_tokens.append(valid_tokens)
                        concept_ids = batch_examples["concept_id"].tolist()
                        concept_strings = []
                        for concept_id in concept_ids:
                            concept_str = f"concept_{concept_id}"
                            for meta_item in self.metadata:
                                if meta_item.get("concept_id") == concept_id:
                                    concept_str = meta_item.get("concept", concept_str)
                                    break
                            concept_strings.append(concept_str)
                        _attn_mask = torch.cat(
                            [
                                inputs["attention_mask"],
                                torch.ones_like(
                                    generations,
                                    dtype=torch.long,
                                    device=inputs["attention_mask"].device,
                                ),
                            ],
                            dim=-1,
                        )
                        self._visualize_token_heatmap(
                            gathered_sparse_mask.cpu().squeeze(),  # move to cpu
                            step=i // batch_size,
                            dump_dir=infer_dump_dir,
                            batch_tokens=batch_tokens,
                            concept_ids=concept_ids,
                            concept_strings=concept_strings,
                            attention_mask=_attn_mask.cpu(),
                            viz_mode="pred/mask",
                            title_prefix="Prediction",
                        )
                        del (
                            batch_tokens,
                            concept_ids,
                            concept_strings,
                            _attn_mask,
                            gathered_sparse_mask,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()

                # Decode and print only the generated text without prompt tokens
                input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
                generated_texts = [
                    self.tokenizer.decode(
                        generation[input_length:], skip_special_tokens=True
                    )
                    for generation, input_length in zip(generations, input_lengths)
                ]
                all_generations += generated_texts

                # Calculate perplexity for each sequence
                unpruned_generated_texts = [
                    self.tokenizer.decode(generation, skip_special_tokens=True)
                    for generation in generations
                ]
                batch_input_ids = self.tokenizer(
                    unpruned_generated_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).input_ids.to(self.device)
                batch_attention_mask = (
                    batch_input_ids != self.tokenizer.pad_token_id
                ).float()

                outputs = self.model(
                    input_ids=batch_input_ids, attention_mask=batch_attention_mask
                )

                logits = outputs.logits[
                    :, :-1, :
                ].contiguous()  # Remove last token prediction
                target_ids = batch_input_ids[:, 1:].contiguous()  # Shift right by 1

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                token_losses = loss_fct(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1)
                )

                token_losses = token_losses.view(batch_input_ids.size(0), -1)
                mask = batch_attention_mask[:, 1:].contiguous()

                seq_lengths = mask.sum(dim=1)
                seq_losses = (token_losses * mask).sum(dim=1) / seq_lengths
                seq_perplexities = torch.exp(seq_losses).cpu().float().tolist()
                all_perplexities.extend(seq_perplexities)
                all_strengths.extend((mag * max_acts).cpu().float().tolist())

                # clear the steering vector generated for this batch
                self.ax._reset_v()
                del v, v_np
                gc.collect()
                torch.cuda.empty_cache()

                # Free up all batch tensors and call empty_cache
                del (
                    batch_examples,
                    input_strings,
                    mag,
                    idx,
                    max_acts,
                    inputs,
                    input_concept,
                    concept_inputs,
                    generations,
                    gen_attention_mask,
                    unpruned_generated_texts,
                    batch_input_ids,
                    batch_attention_mask,
                    outputs,
                    logits,
                    target_ids,
                    loss_fct,
                    token_losses,
                    mask,
                    seq_lengths,
                    seq_losses,
                    seq_perplexities,
                    generated_texts,
                    subspaces,
                    locs,
                )
                gc.collect()
                torch.cuda.empty_cache()

        return {
            "steered_generation": all_generations,
            "perplexity": all_perplexities,
            "strength": all_strengths,
            "steering_vector": all_steering_vectors,
        }

    def _logit_diff_visualization(
        self,
        base_output,
        cf_outputs,
        inputs,
        normalize=False,
        mode="train",
        step=None,
        dump_dir=None,
    ):
        if not self.model_config.logit_diff_visualization.log_heatmap:
            return

        full_base_logits = base_output.logits
        full_cf_logits = cf_outputs.logits

        # Compute logprobs and logit diff
        cf_logps = F.log_softmax(full_cf_logits, dim=-1)
        base_logps = F.log_softmax(full_base_logits, dim=-1)
        logit_diff = cf_logps - base_logps
        logit_diff = (logit_diff * inputs["attention_mask"].unsqueeze(-1)).sum(dim=-1)

        if normalize:
            logit_diff = logit_diff / inputs["attention_mask"].sum(dim=-1)

        # Clean mode string for directory name
        mode_dir = str(mode).replace("/", "_").replace(" ", "_")
        dump_dir_final = os.path.join(dump_dir, mode_dir)

        viz_mode_str = f"{mode}/logit_diff" if mode else "train/logit_diff"
        self._visualize_token_heatmap(
            logit_diff,
            step=step,
            dump_dir=dump_dir_final,
            batch_tokens=[
                self.tokenizer.convert_ids_to_tokens(input_id)
                for input_id in inputs["input_ids"]
            ],
            attention_mask=inputs["attention_mask"],
            concept_ids=inputs["concept_ids"] if "concept_ids" in inputs else None,
            concept_strings=[
                f"concept_{concept_id}" for concept_id in inputs["concept_ids"].tolist()
            ]
            if "concept_ids" in inputs and hasattr(inputs["concept_ids"], "tolist")
            else None,
            viz_mode=viz_mode_str,
            title_prefix="Logit Diff",
        )

    def _save_cross_attn_heatmaps(
        self,
        attn_weights,
        input_ids,
        concept_ids,
        input_attention_mask,
        concept_attention_mask,
        dump_dir,
        batch_idx,
        tokenizer=None,
        prefix="cross_attn",
        max_samples=5,  # Only visualize up to 5 random samples per batch
    ):
        """
        Save cross-attention heatmap grids for a random subset of samples in a batch.
        Each grid: rows=heads, columns=layers, each cell is a heatmap.
        attn_weights: list/tuple of (num_layers,) each [batch, num_heads, q_len, kv_len]
        input_ids: [batch, seq_len] (input tokens)
        concept_ids: [batch, seq_len] (concept tokens)
        input_attention_mask: [batch, seq_len] (mask for input tokens)
        concept_attention_mask: [batch, seq_len] (mask for concept tokens)
        dump_dir: directory to save PNGs
        batch_idx: int, batch number
        tokenizer: tokenizer to decode tokens
        prefix: filename prefix
        max_samples: maximum number of samples to visualize per batch
        """

        os.makedirs(dump_dir, exist_ok=True)
        if tokenizer is None:
            tokenizer = self.tokenizer
        clean = self.visualizer.clean_text_for_display

        num_layers = len(attn_weights)
        if num_layers == 0:
            return

        num_heads = attn_weights[0].shape[1]
        batch_size = attn_weights[0].shape[0]
        # Pick up to max_samples random indices
        if batch_size > max_samples:
            sample_indices = random.sample(range(batch_size), max_samples)
        else:
            sample_indices = list(range(batch_size))
        for sample_idx in sample_indices:
            # Create a subdirectory for this sample
            sample_dir = os.path.join(dump_dir, f"sample_{sample_idx}")
            os.makedirs(sample_dir, exist_ok=True)
            input_mask = input_attention_mask[sample_idx].detach().cpu().bool().numpy()
            input_tokens = [
                t
                for t, m in zip(
                    tokenizer.convert_ids_to_tokens(
                        input_ids[sample_idx].detach().cpu()
                    ),
                    input_mask,
                )
                if m
            ]
            input_tokens = clean(input_tokens)
            # Decode concept string for this sample (without special tokens)
            concept_str = tokenizer.decode(
                concept_ids[sample_idx].detach().cpu(),
                skip_special_tokens=True,
            )
            # Truncate or wrap concept string for title
            max_title_len = 80
            if len(concept_str) > max_title_len:
                concept_str_disp = concept_str[:max_title_len] + "..."
            else:
                concept_str_disp = concept_str
            for l in range(num_layers):
                attn_layer = attn_weights[l][sample_idx]  # [num_heads, q_len, kv_len]
                # Get valid (unpadded) tokens for axes using indices
                input_mask = (
                    input_attention_mask[sample_idx].detach().cpu().bool().numpy()
                )
                input_indices = np.where(input_mask)[0]
                input_tokens = [
                    t
                    for i, t in enumerate(
                        tokenizer.convert_ids_to_tokens(
                            input_ids[sample_idx].detach().cpu()
                        )
                    )
                    if input_mask[i]
                ]

                # Replace each whitespace token in input_tokens with '[SPACE]'
                input_tokens = [
                    "[SPACE]" if t == "" or t == "\n" or t == " " else t
                    for t in input_tokens
                ]

                concept_mask = (
                    concept_attention_mask[sample_idx].detach().cpu().bool().numpy()
                )
                concept_indices = np.where(concept_mask)[0]
                concept_tokens = [
                    t
                    for i, t in enumerate(
                        tokenizer.convert_ids_to_tokens(
                            concept_ids[sample_idx].detach().cpu()
                        )
                    )
                    if concept_mask[i]
                ]
                concept_tokens = clean(concept_tokens)
                num_heads, q_len, kv_len = attn_layer.shape
                fig, axes = plt.subplots(
                    nrows=num_heads,
                    ncols=1,
                    figsize=(max(6, len(input_tokens) // 2), max(3, num_heads * 2)),
                    sharex=True,
                )
                if num_heads == 1:
                    axes = [axes]
                im = None
                for h in range(num_heads):
                    ax = axes[h]
                    # Use np.ix_ to select the correct submatrix
                    attn_2d = (
                        attn_layer[h][np.ix_(concept_indices, input_indices)]
                        .cpu()
                        .float()
                        .numpy()
                    )
                    im = ax.imshow(
                        attn_2d, aspect="auto", cmap="viridis", vmin=0, vmax=1
                    )
                    ax.set_ylabel(f"Head {h}")
                    ax.set_yticks(np.arange(len(concept_tokens)))
                    ax.set_yticklabels(concept_tokens, fontsize=6)
                    if h == num_heads - 1:
                        ax.set_xticks(np.arange(len(input_tokens)))
                        ax.set_xticklabels(input_tokens, rotation=90, fontsize=6)
                        ax.set_xlabel("Input Tokens")
                    else:
                        ax.set_xticks([])
                fig.suptitle(
                    f"{prefix} Layer {l} | Concept: {concept_str_disp}", fontsize=10
                )
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                if im is not None:
                    fig.colorbar(im, ax=axes, fraction=0.02)
                fname = os.path.join(sample_dir, f"layer_{l}.png")
                plt.savefig(fname, dpi=100)
                plt.close(fig)
