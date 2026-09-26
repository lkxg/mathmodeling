"""Load the validation-selected contextual fusion model or ensemble."""
from __future__ import annotations

import json

import torch
from torch import nn

from .core import OUT
from .fusion_variants import VARIANTS


class SelectedEnsemble(nn.Module):
    def __init__(self, names, device):
        super().__init__()
        models=[]
        for name in names:
            checkpoint=torch.load(OUT/"experiments"/f"{name}.pt",map_location="cpu",weights_only=True)
            model=VARIANTS[checkpoint["name"]]()
            model.load_state_dict(checkpoint["state_dict"],strict=True)
            models.append(model.to(device).eval())
        self.models=nn.ModuleList(models)

    def forward(self,batch):
        out=[model(batch) for model in self.models]
        return {"logits":torch.stack([o["logits"] for o in out]).mean(0),
                "reg":torch.stack([o["reg"] for o in out]).mean(0),
                "representation":torch.stack([o["representation"] for o in out]).mean(0)}


def load_selected(device):
    choice=json.loads((OUT/"experiments"/"selection.json").read_text())
    return SelectedEnsemble(choice["selected_models"],device).eval(),choice
