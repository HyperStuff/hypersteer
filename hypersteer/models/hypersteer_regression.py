import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from hypersteer.utils.helpers import (
    configure_tokenizer_model,
    get_logger,
    set_default_device,
)

from .hypersteer_base import HyperSteerBase

logger = get_logger(__name__)


class RegressionWrapper(nn.Module):
    """Wraps a base model with a regression head for concept embedding."""

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


class HyperSteerRegression(HyperSteerBase):
    """HyperSteer with regression-based concept embedding."""

    def __str__(self):
        return "HyperSteerRegression"

    def _create_concept_embedding(self) -> tuple[nn.Module, AutoTokenizer]:
        """Create regression-based concept embedding."""
        base_model_tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model_name, model_max_length=512
        )
        base_model_tokenizer.padding_side = "left"

        with set_default_device(self.device):
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_config.base_model_name, torch_dtype=torch.bfloat16
            )
            configure_tokenizer_model(base_model, base_model_tokenizer)
            concept_embedding = RegressionWrapper(
                base_model=base_model,
                hidden_size=base_model.config.hidden_size,
                output_dim=self.model.config.hidden_size,
            )

        return concept_embedding, base_model_tokenizer

    def _compute_concept_embedding(
        self, inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute concept embedding using regression model."""
        return self.concept_embedding(
            inputs["concept_input_ids"],
            inputs["concept_attention_mask"],
        )

    def _compute_concept_embedding_for_inference(
        self, inputs: dict, concept_inputs: dict
    ) -> torch.Tensor:
        """Compute concept embedding for inference using regression model."""
        return self.concept_embedding(
            concept_inputs["input_ids"],
            concept_inputs["attention_mask"],
        )

    def get_logits(self, concept_id, k=10):
        """Get top logits for a concept (regression-specific)."""
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

        concept_text = self.concept_id_to_text.get(concept_id)
        if concept_text is None:
            raise ValueError(f"Concept ID {concept_id} not found in concept mapping.")

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
