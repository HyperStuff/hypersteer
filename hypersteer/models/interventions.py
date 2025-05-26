from dataclasses import dataclass
from typing import Any

import torch
from pyvene import (
    DistributedRepresentationIntervention,
    InterventionOutput,
    SourcelessIntervention,
    TrainableIntervention,
)
from torch import nn
from torch.nn import functional as F


@dataclass
class PayloadInterventionOutput(InterventionOutput):
    """
    Output of the IntervenableModel, including original outputs, intervened outputs, and collected activations.
    """

    output: Any | None = None
    latent: Any | None = None
    payload: Any | None = None


class SelectionHead(nn.Module):
    def __init__(self, hidden_size: int, use_ln: bool = True):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 2, 1)
        self.ln = None
        if use_ln:
            self.ln = nn.LayerNorm((hidden_size * 2,))

    def forward(self, x, v):
        latent = torch.cat([x, v.unsqueeze(1).expand_as(x)], dim=-1)
        if self.ln:
            latent = self.ln(latent)
        return F.sigmoid(self.proj(latent))


class HyperAdditiveIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__(**kwargs, keep_last_dim=True)
        self.low_rank_dimension = kwargs["low_rank_dimension"]
        self.v: torch.Tensor = None
        self.use_selection = kwargs.get("use_selection_head", False)

        if self.use_selection:
            self.selection_head = SelectionHead(
                self.embed_dim, use_ln=kwargs.get("use_ln", True)
            )

    def _update_v(self, new_vect: torch.Tensor):
        self.v = new_vect

    def _reset_v(self):
        self.v = None

    def forward(self, base, source=None, subspaces=None):
        mag = subspaces["mag"][:, None, None] if subspaces and "mag" in subspaces else 1
        mask = self.selection_head(base, self.v) if self.use_selection else 1

        output = base + mask * mag * self.v.unsqueeze(dim=1)

        return PayloadInterventionOutput(
            output=output,
            payload={
                "mask": mask,
            },
        )
