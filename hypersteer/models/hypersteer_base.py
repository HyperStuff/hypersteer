import gc
import os
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from pyvene import IntervenableConfig, IntervenableModel
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hypersteer.data.utils import get_batch_locs, make_data_module
from hypersteer.training import TrainerMixin
from hypersteer.utils.debug_utils import debug_print
from hypersteer.utils.helpers import get_logger
from hypersteer.utils.model_utils import calculate_perplexity
from hypersteer.utils.visualization import Visualizer

from .model import Model
from .modules.interventions import HyperAdditiveIntervention

logger = get_logger(__name__)


class HyperSteerBase(Model, TrainerMixin):
    """Base HyperSteer model with shared logic. Subclass for specific hypernet types."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visualizer = Visualizer()
        self.concept_embedding: nn.Module = None
        self.base_model_tokenizer = None

    @abstractmethod
    def _create_concept_embedding(self) -> nn.Module:
        """Create and return the concept embedding module. Implemented by subclasses."""
        pass

    @abstractmethod
    def _compute_concept_embedding(
        self, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute concept embedding vector from inputs. Implemented by subclasses."""
        pass

    def make_model(self, **kwargs):
        self.ax = HyperAdditiveIntervention(
            embed_dim=self.model.config.hidden_size,
            low_rank_dimension=self.model_config.low_rank_dimension,
            use_selection_head=self.model_config.use_selection_head,
            use_ln=self.model_config.use_selection_ln,
            selection_head_start_temperature=self.model_config.selection_head_start_temperature,
            selection_head_end_temperature=self.model_config.selection_head_end_temperature,
            selection_head_learnable_temperature=self.model_config.selection_head_learnable_temperature,
            selection_head_anneal_temperature=self.model_config.selection_head_anneal_temperature,
            selection_head_add_gumbel_noise=self.model_config.selection_head_add_gumbel_noise,
            selection_head_threshold=self.model_config.selection_head_threshold,
            selection_head_straight_through=self.model_config.selection_head_straight_through,
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

        # Create concept embedding (implemented by subclass)
        self.concept_embedding, self.base_model_tokenizer = (
            self._create_concept_embedding()
        )
        self.concept_embedding = self.concept_embedding.to(torch.bfloat16)

        # Initialize empty concept mapping - will be populated from dataset
        self.concept_id_to_text = {}

        self.eval_steps = kwargs.get("eval_steps", 20)
        self.log_per_step = kwargs.get("log_per_step", 10)
        self.include_sentence_in_embedding = kwargs.get(
            "include_sentence_in_embedding", False
        )

    def setup_model(self):
        """Setup the model for training."""
        self.concept_embedding.train()
        self.ax.train()

    def setup_optimizer(self):
        """Setup and return the optimizer."""
        _param_groups = [
            {
                "params": self.concept_embedding.parameters(),
                "lr": self.training_args.lr,
            },
        ]

        if self.model_config.use_selection_head:
            _param_groups.append(
                {
                    "params": [
                        p for n, p in self.ax.named_parameters() if "temperature" in n
                    ],
                    "lr": self.model_config.temperature_lr,
                }
            )
            _param_groups.append(
                {
                    "params": [
                        p
                        for n, p in self.ax.named_parameters()
                        if "temperature" not in n
                    ],
                    "lr": self.training_args.lr,
                }
            )

        optimizer = torch.optim.AdamW(
            _param_groups, weight_decay=self.training_args.weight_decay
        )
        return optimizer

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return parameters that should be included in gradient clipping."""
        return list(self.ax.parameters()) + list(self.concept_embedding.parameters())

    def post_backward(
        self,
        step_outputs,
        lr_scheduler,
        optimizer,
        global_step,
    ) -> dict[str, Any]:
        # Gradient clipping
        ax_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.ax.parameters(),
            self.training_args.max_grad_norm,
        )
        concept_embedding_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.concept_embedding.parameters(),
            self.training_args.max_grad_norm,
        )
        _all_params = list(self.ax.parameters()) + list(
            self.concept_embedding.parameters()
        )
        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            _all_params,
            self.training_args.max_grad_norm,
        )
        step_outputs[("grad_norms", "ax")] = ax_grad_norm
        step_outputs[("grad_norms", "concept_embedding")] = concept_embedding_grad_norm
        step_outputs[("grad_norms", "total")] = total_grad_norm
        step_outputs[("lr", "main")] = lr_scheduler.get_last_lr()[0]
        if self.model_config.selection_head_learnable_temperature:
            step_outputs[("lr", "temperature")] = lr_scheduler.get_last_lr()[-1]

        if (
            self.model_config.selection_head_anneal_temperature
            and not self.model_config.selection_head_learnable_temperature
        ):
            self.ax.selection_head.step_temperature(global_step)

        return step_outputs

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

        step_outputs = {}

        # Add selection sparsity loss if enabled
        if self.model_config.use_selection_head:
            mask = self.ax_model.full_intervention_outputs[0].payload["mask"]
            with (
                nullcontext()
                if self.model_config.compute_sparsity_loss
                else torch.no_grad()
            ):
                selection_sparsity_loss = self.compute_selection_sparsity_loss(
                    gathered_sparse_mask=mask,
                    attention_mask=inputs["attention_mask"],
                    locs=inputs["intervention_locations"],
                )
            if self.model_config.compute_sparsity_loss:
                loss += (
                    selection_sparsity_loss * self.model_config.selection_l1_loss_coeff
                )
                step_outputs[("loss", "mask_l1")] = selection_sparsity_loss

            with torch.no_grad():
                step_outputs[("metrics", "mask_sparsity")] = 1 - selection_sparsity_loss

            # Visualize sparse mask at the configured frequency
            self._visualize_training_mask(mask, inputs, global_step)

        step_outputs[("loss", "main")] = loss

        return step_outputs

    @torch.no_grad()
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

        # Compute concept embedding (delegated to subclass)
        v = self._compute_concept_embedding(inputs)

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
        step_outputs = {("loss", "main"): steering_loss}

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
            step_outputs[("loss", "mask_l1")] = selection_sparsity_loss

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

    def on_validation_start(self):
        """Called at the start of validation."""
        self.concept_embedding.eval()
        self.ax.eval()

    def on_validation_end(self):
        """Called at the end of validation."""
        self.concept_embedding.train()
        self.ax.train()

    def _visualize_training_mask(self, mask, inputs, global_step):
        """Visualize training mask."""
        batch_tokens = [
            self.tokenizer.convert_ids_to_tokens(input_id)
            for input_id in inputs["input_ids"]
        ]
        concept_strings = []
        for concept_id in inputs["concept_ids"]:
            concept_id_val = concept_id.item()
            concept_str = self.concept_id_to_text.get(
                concept_id_val, f"concept_{concept_id_val}"
            )
            concept_strings.append(concept_str)

        self._visualize_token_heatmap(
            mask.squeeze(-1),
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
            concept_id_val = concept_id.item()
            concept_str = self.concept_id_to_text.get(
                concept_id_val, f"concept_{concept_id_val}"
            )
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

    def compute_main_loss_and_outputs(
        self,
        inputs,
        unit_locations,
        subspaces,
    ):
        # Compute concept embedding (delegated to subclass)
        v = self._compute_concept_embedding(inputs)

        if self.model_config.debug_print:
            debug_print(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs["labels"],
                inputs["intervention_locations"],
                inputs["concept_input_ids"],
                self.tokenizer,
            )

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
        """Helper to visualize sparse mask using the configured visualization options."""
        if not self.model_config.mask_visualization.log_heatmap:
            return
        freq = self.model_config.mask_visualization.log_heatmap_freq
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

    def _extract_concept_metadata_from_dataset(self, examples):
        """Extract concept ID to concept text mapping from the dataset."""
        concept_id_to_text = {}

        if "output_concept" in examples.columns:
            for _, row in examples.iterrows():
                concept_id = row.get("concept_id")
                concept_text = row.get("output_concept")
                if concept_id is not None and concept_text is not None:
                    concept_id_to_text[concept_id] = concept_text
        elif "input_concept" in examples.columns:
            for _, row in examples.iterrows():
                concept_id = row.get("concept_id")
                concept_text = row.get("input_concept")
                if concept_id is not None and concept_text is not None:
                    concept_id_to_text[concept_id] = concept_text

        return concept_id_to_text

    def make_dataloader(
        self, examples, rank=0, world_size=1, shuffle=True, distributed=False, **kwargs
    ):
        # Extract concept metadata from dataset before creating dataloader
        extracted_concept_mapping = self._extract_concept_metadata_from_dataset(
            examples
        )
        self.concept_id_to_text.update(extracted_concept_mapping)

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
        weight_file = os.path.join(dump_dir, f"{model_name}_weight.safetensors")
        self.concept_embedding.cpu()
        # Only save trainable parameters (exclude base model weights which are redundant)
        trainable_state_dict = {
            k: v.clone() 
            for k, v in self.concept_embedding.named_parameters() 
            if v.requires_grad
        }
        save_file(trainable_state_dict, weight_file)

        if hasattr(self.ax, "selection_head") and self.model_config.use_selection_head:
            path = os.path.join(dump_dir, f"{model_name}_selection_head.safetensors")
            # Clone tensors to avoid shared memory issues
            selection_state_dict = {k: v.clone() for k, v in self.ax.selection_head.state_dict().items()}
            save_file(selection_state_dict, path)
            logger.debug(f"Saved selection head to {path}")

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = os.path.join(dump_dir, f"{model_name}_weight.safetensors")
        self.make_model(**kwargs)

        # Load only trainable parameters (base model weights are not saved)
        # Note: safetensors doesn't support MPS, so load to CPU first
        load_device = "cpu" if "mps" in str(self.device) else str(self.device)
        saved_state_dict = load_file(weight_file, device=load_device)
        # Filter to only load parameters that exist and are trainable
        concept_embedding_state_dict = self.concept_embedding.state_dict()
        filtered_state_dict = {
            k: v for k, v in saved_state_dict.items() 
            if k in concept_embedding_state_dict
        }
        self.concept_embedding.load_state_dict(filtered_state_dict, strict=False)
        self.concept_embedding.to(self.device)

        if self.model_config.use_selection_head and hasattr(self.ax, "selection_head"):
            path = os.path.join(dump_dir, f"{model_name}_selection_head.safetensors")
            if os.path.exists(path):
                self.ax.selection_head.load_state_dict(
                    load_file(path, device=load_device)
                )
                self.ax.selection_head.to(self.device)
                logger.debug(f"Loaded selection head from {path}")

    def predict_step(self, batch_examples, batch_idx, **kwargs):
        """Prediction step with concept embeddings and visualizations."""
        self.dump_dir = kwargs.get("dump_dir", None)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)

        infer_dump_dir = os.path.join(
            kwargs.get("dump_dir") or "assets/cache/sparse_masks", "inference_steer"
        )
        os.makedirs(infer_dump_dir, exist_ok=True)

        input_strings = batch_examples["input"].tolist()

        mag = torch.tensor(batch_examples["factor"].tolist()).to(self.device)
        idx = torch.tensor(batch_examples["concept_id"].tolist()).to(self.device)

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

        # Compute concept embedding (delegated to subclass)
        v = self._compute_concept_embedding_for_inference(inputs, concept_inputs)

        # Store steering vectors
        v_np = v.detach().float().cpu().numpy()
        steering_vectors = [row.copy() for row in v_np]

        self.ax._update_v(v)

        locs = get_batch_locs(
            prefix_length=kwargs["prefix_length"],
            inputs=batch_examples["input"],
            attention_mask=inputs["attention_mask"],
            tokenizer=self.tokenizer,
        )

        subspaces = [
            {
                "idx": idx,
                "mag": mag,
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

        if isinstance(steered_out, dict):
            generations = steered_out.get("sequences", None)
        else:
            generations = getattr(steered_out, "sequences", None)

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

        if (
            self.model_config.logit_diff_visualization.log_heatmap
            and batch_idx % self.model_config.logit_diff_visualization.log_heatmap_freq
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

        input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
        generated_texts = [
            self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
            for generation, input_length in zip(generations, input_lengths)
        ]

        unpruned_generated_texts = [
            self.tokenizer.decode(generation, skip_special_tokens=True)
            for generation in generations
        ]

        perplexities = calculate_perplexity(
            self.model, self.tokenizer, unpruned_generated_texts, self.device
        )

        self.ax._reset_v()

        del base_out, steered_out, base_out_gen, steered_out_gen, v, v_np
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "generations": generated_texts,
            "perplexities": perplexities,
            "steering_vectors": steering_vectors,
        }

    @abstractmethod
    def _compute_concept_embedding_for_inference(
        self, inputs: dict, concept_inputs: dict
    ) -> torch.Tensor:
        """Compute concept embedding for inference. Implemented by subclasses."""
        pass

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

        cf_logps = torch.nn.functional.log_softmax(full_cf_logits, dim=-1)
        base_logps = torch.nn.functional.log_softmax(full_base_logits, dim=-1)
        logit_diff = cf_logps - base_logps
        logit_diff = (logit_diff * inputs["attention_mask"].unsqueeze(-1)).sum(dim=-1)

        if normalize:
            logit_diff = logit_diff / inputs["attention_mask"].sum(dim=-1)

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
                self.concept_id_to_text.get(concept_id, f"concept_{concept_id}")
                for concept_id in inputs["concept_ids"].tolist()
            ]
            if "concept_ids" in inputs and hasattr(inputs["concept_ids"], "tolist")
            else None,
            viz_mode=viz_mode_str,
            title_prefix="Logit Diff",
        )
