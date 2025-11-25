import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from hypersteer.utils.helpers import get_logger, set_default_device

from .hypernet.configuration_hypernet import HypernetConfig
from .hypernet.modeling_hypernet import HypernetModel
from .hypersteer_base import HyperSteerBase

logger = get_logger(__name__)


class HyperSteerAttn(HyperSteerBase):
    """HyperSteer with cross-attention based concept embedding."""

    def __str__(self):
        return "HyperSteerAttn"

    def _create_concept_embedding(self) -> tuple[nn.Module, AutoTokenizer]:
        """Create cross-attention based concept embedding (HypernetModel)."""
        base_model_tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_name, model_max_length=512
        )
        base_model_tokenizer.padding_side = "left"

        base_model_config = AutoConfig.from_pretrained(
            self.model_config.base_model_name
        )

        hypernet_config = HypernetConfig(
            num_hidden_layers=self.model_config.cross_attn_hidden_layers,
            target_model_name_or_path=self.model_config.base_model_name,
            hidden_size=base_model_config.hidden_size,
            torch_dtype=torch.bfloat16,
        )

        with set_default_device(self.device):
            concept_embedding = HypernetModel(config=hypernet_config)

        return concept_embedding, base_model_tokenizer

    def _compute_concept_embedding(
        self, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute concept embedding using cross-attention model."""
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

        return v

    def _compute_concept_embedding_for_inference(
        self, inputs: dict, concept_inputs: dict
    ) -> torch.Tensor:
        """Compute concept embedding for inference using cross-attention model."""
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
            # Cross attention weights could be logged here if needed
            return v
        else:
            return self.concept_embedding(
                input_ids=None,
                inputs_embeds=concept_inputs_embeds,
                attention_mask=concept_inputs["attention_mask"],
                base_encoder_hidden_states=base_hidden_state,
                base_encoder_attention_mask=base_intervention_mask,
                output_hidden_states=False,
            ).last_hidden_state

    def predict_step(self, batch_examples, batch_idx, **kwargs):
        """Override predict_step to add cross-attention visualization."""
        # Setup directories for cross-attention heatmaps
        cross_attn_dump_dir = os.path.join(
            kwargs.get("dump_dir") or "assets/cache/sparse_masks", "cross_attn_heatmaps"
        )
        os.makedirs(cross_attn_dump_dir, exist_ok=True)

        # Check if we need to visualize cross-attention
        if (
            self.model_config.cross_attn_heatmap_visualization.log_heatmap
            and batch_idx
            % self.model_config.cross_attn_heatmap_visualization.log_heatmap_freq
            == 0
        ):
            # Do inference with attention outputs for visualization
            result = self._predict_step_with_cross_attn_viz(
                batch_examples, batch_idx, cross_attn_dump_dir, **kwargs
            )
            return result
        else:
            # Use base class implementation
            return super().predict_step(batch_examples, batch_idx, **kwargs)

    def _predict_step_with_cross_attn_viz(
        self, batch_examples, batch_idx, cross_attn_dump_dir, **kwargs
    ):
        """Prediction step with cross-attention visualization."""
        import gc

        from hypersteer.data.utils import get_batch_locs
        from hypersteer.utils.model_utils import calculate_perplexity

        self.dump_dir = kwargs.get("dump_dir", None)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)

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

        # Compute with cross-attention visualization
        concept_inputs_embeds = self.model.model.embed_tokens(
            concept_inputs["input_ids"]
        )
        base_intervention_mask = inputs["attention_mask"]
        base_hidden_state = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        ).hidden_states[self.layer]

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

        # Save cross-attention heatmaps
        self._save_cross_attn_heatmaps(
            cross_attn_weights,
            inputs["input_ids"],
            concept_inputs["input_ids"],
            inputs["attention_mask"],
            concept_inputs["attention_mask"],
            cross_attn_dump_dir,
            batch_idx,
            tokenizer=self.tokenizer,
            prefix="cross_attn",
        )

        # Continue with standard prediction flow
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
                {"input_ids": generations, "attention_mask": gen_attention_mask},
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
        max_samples=5,
    ):
        """Save cross-attention heatmap grids for a random subset of samples."""
        os.makedirs(dump_dir, exist_ok=True)
        if tokenizer is None:
            tokenizer = self.tokenizer
        clean = self.visualizer.clean_text_for_display

        num_layers = len(attn_weights)
        if num_layers == 0:
            return

        num_heads = attn_weights[0].shape[1]
        batch_size = attn_weights[0].shape[0]

        if batch_size > max_samples:
            sample_indices = random.sample(range(batch_size), max_samples)
        else:
            sample_indices = list(range(batch_size))

        for sample_idx in sample_indices:
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

            concept_str = tokenizer.decode(
                concept_ids[sample_idx].detach().cpu(),
                skip_special_tokens=True,
            )
            max_title_len = 80
            if len(concept_str) > max_title_len:
                concept_str_disp = concept_str[:max_title_len] + "..."
            else:
                concept_str_disp = concept_str

            for layer_idx in range(num_layers):
                attn_layer = attn_weights[layer_idx][sample_idx]

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

                num_heads_layer, q_len, kv_len = attn_layer.shape
                fig, axes = plt.subplots(
                    nrows=num_heads_layer,
                    ncols=1,
                    figsize=(max(6, len(input_tokens) // 2), max(3, num_heads_layer * 2)),
                    sharex=True,
                )
                if num_heads_layer == 1:
                    axes = [axes]

                im = None
                for h in range(num_heads_layer):
                    ax = axes[h]
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
                    if h == num_heads_layer - 1:
                        ax.set_xticks(np.arange(len(input_tokens)))
                        ax.set_xticklabels(input_tokens, rotation=90, fontsize=6)
                        ax.set_xlabel("Input Tokens")
                    else:
                        ax.set_xticks([])

                fig.suptitle(
                    f"{prefix} Layer {layer_idx} | Concept: {concept_str_disp}",
                    fontsize=10,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                if im is not None:
                    fig.colorbar(im, ax=axes, fraction=0.02)

                fname = os.path.join(sample_dir, f"layer_{layer_idx}.png")
                plt.savefig(fname, dpi=100)
                plt.close(fig)
