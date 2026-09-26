"""Alternative fusion architectures evaluated against the original q23 model.

They share the exact same BERT contextual text feature interface. References
in the accompanying report describe inspirations, not literal reproductions.
"""
from __future__ import annotations

import torch
from torch import nn


class VariantBase(nn.Module):
    def __init__(self, hidden=128, dropout=.3, layers=1):
        super().__init__()
        self.hidden = hidden
        self.position = nn.Embedding(50, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden*2, dropout,
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 4))
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                                    nn.Linear(hidden, 4)) for _ in range(3)])

    def finish(self, fused, local, batch):
        mask, valid = batch["mask"], batch["valid"]
        n, length = valid.shape
        fused = fused + self.position(torch.arange(length, device=valid.device))[None]
        fused = self.temporal(fused, src_key_padding_mask=~valid)
        weights = self.pool(fused).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)
        pooled = (fused * weights[..., None]).sum(1)
        raw = self.head(pooled)
        summaries = [local[m].sum(1) / mask[...,m].sum(1, keepdim=True).clamp_min(1) for m in range(3)]
        aux = torch.stack([self.experts[m](summaries[m]) for m in range(3)], dim=1)
        return {"logits": raw[:,:3], "reg": 3*torch.tanh(raw[:,3]/3),
                "aux": aux, "availability": mask.any(1), "pool": weights,
                "representation": pooled}


class TextAnchor(VariantBase):
    """Text-centred aligned residual fusion, with A/V fallback if text is absent."""
    def __init__(self):
        super().__init__(layers=2)
        self.project = nn.ModuleList([nn.Sequential(nn.Linear(d,128),nn.LayerNorm(128),nn.GELU())
                                      for d in (768,74,35)])
        self.gate = nn.Sequential(nn.Linear(128*3+3,128),nn.GELU(),nn.Linear(128,3))
        self.combine = nn.Sequential(nn.Linear(128*2,128),nn.LayerNorm(128),nn.GELU())

    def forward(self,batch):
        mask = batch["mask"]
        local = [self.project[m](batch[k])*mask[...,m,None]
                 for m,k in enumerate(("text","audio","vision"))]
        score = self.gate(torch.cat(local+[mask.float()],dim=-1)).masked_fill(~mask,-1e4)
        gate = score.softmax(-1)*mask.float()
        gate = gate/gate.sum(-1,keepdim=True).clamp_min(1e-8)
        proxy = sum(gate[...,m,None]*local[m] for m in range(3))
        fused = self.combine(torch.cat([proxy,local[0]],dim=-1))
        return self.finish(fused,local,batch)


class SharedPrivate(VariantBase):
    """Shared sentiment channel plus separately retained modal details."""
    def __init__(self):
        super().__init__(layers=1)
        self.shared = nn.ModuleList([nn.Linear(d,128) for d in (768,74,35)])
        self.private = nn.ModuleList([nn.Linear(d,48) for d in (768,74,35)])
        self.combine = nn.Sequential(nn.Linear(128+48*3+3,128),nn.LayerNorm(128),nn.GELU())

    def forward(self,batch):
        mask=batch["mask"]
        values=[batch[k] for k in ("text","audio","vision")]
        shared=[torch.nn.functional.gelu(self.shared[m](values[m]))*mask[...,m,None] for m in range(3)]
        private=[torch.nn.functional.gelu(self.private[m](values[m]))*mask[...,m,None] for m in range(3)]
        denom=mask.float().sum(-1,keepdim=True).clamp_min(1)
        common=sum(shared)/denom
        fused=self.combine(torch.cat([common,*private,mask.float()],dim=-1))
        return self.finish(fused,shared,batch)


class ReliabilityProxy(VariantBase):
    """Learned uncertainty weights combine available modal features."""
    def __init__(self):
        super().__init__(layers=1)
        self.mean=nn.ModuleList([nn.Linear(d,128) for d in (768,74,35)])
        self.log_variance=nn.ModuleList([nn.Linear(d,1) for d in (768,74,35)])
        self.private=nn.ModuleList([nn.Linear(d,32) for d in (768,74,35)])
        self.combine=nn.Sequential(nn.Linear(128+32*3+3,128),nn.LayerNorm(128),nn.GELU())

    def forward(self,batch):
        mask=batch["mask"]
        values=[batch[k] for k in ("text","audio","vision")]
        means=[torch.nn.functional.gelu(self.mean[m](values[m]))*mask[...,m,None] for m in range(3)]
        scores=torch.cat([-self.log_variance[m](values[m]).clamp(-4,4) for m in range(3)],dim=-1)
        scores=scores.masked_fill(~mask,-1e4)
        weights=scores.softmax(-1)*mask.float()
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        proxy=sum(weights[...,m,None]*means[m] for m in range(3))
        details=[torch.nn.functional.gelu(self.private[m](values[m]))*mask[...,m,None] for m in range(3)]
        fused=self.combine(torch.cat([proxy,*details,mask.float()],dim=-1))
        return self.finish(fused,means,batch)


class DualQueryFusion(VariantBase):
    """Unimodal learned queries followed by one cross-modal learned query."""
    def __init__(self):
        super().__init__(layers=1)
        self.project=nn.ModuleList([nn.Sequential(nn.Linear(d,128),nn.LayerNorm(128),nn.GELU())
                                    for d in (768,74,35)])
        self.unimodal_queries=nn.Parameter(torch.randn(3,128)*.02)
        self.cross_query=nn.Parameter(torch.randn(128)*.02)
        self.local_gate=nn.Sequential(nn.Linear(128*3+3,128),nn.GELU(),nn.Linear(128,3))
        self.combine=nn.Sequential(nn.Linear(256,128),nn.LayerNorm(128),nn.GELU())

    @staticmethod
    def query_pool(query,values,mask):
        score=(values*query).sum(-1)/(values.shape[-1]**.5)
        weight=score.masked_fill(~mask,-1e4).softmax(-1)*mask.float()
        weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
        return (values*weight[...,None]).sum(-2)

    def forward(self,batch):
        mask=batch["mask"]
        local=[self.project[m](batch[k])*mask[...,m,None]
               for m,k in enumerate(("text","audio","vision"))]
        unimodal=torch.stack([self.query_pool(self.unimodal_queries[m],local[m],mask[...,m])
                              for m in range(3)],dim=1)
        global_context=self.query_pool(self.cross_query,unimodal,mask.any(1))
        score=self.local_gate(torch.cat(local+[mask.float()],dim=-1)).masked_fill(~mask,-1e4)
        weights=score.softmax(-1)*mask.float()
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        aligned=sum(weights[...,m,None]*local[m] for m in range(3))
        context=global_context[:,None,:].expand_as(aligned)
        fused=self.combine(torch.cat([aligned,context],dim=-1))
        return self.finish(fused,local,batch)


class EnhanceBalance(VariantBase):
    """Cross-modal compensation then sample- and position-aware trust fusion."""
    def __init__(self):
        super().__init__(layers=1)
        self.project=nn.ModuleList([nn.Sequential(nn.Linear(d,128),nn.LayerNorm(128),nn.GELU())
                                    for d in (768,74,35)])
        self.compensate=nn.ModuleList([nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,128))
                                       for _ in range(3)])
        self.trust=nn.ModuleList([nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
                                  for _ in range(3)])
        self.combine=nn.Sequential(nn.Linear(256,128),nn.LayerNorm(128),nn.GELU())

    def forward(self,batch):
        mask=batch["mask"]
        local=[self.project[m](batch[k])*mask[...,m,None]
               for m,k in enumerate(("text","audio","vision"))]
        common=sum(local)/mask.float().sum(-1,keepdim=True).clamp_min(1)
        enhanced=[]
        scores=[]
        for m in range(3):
            supplement=self.compensate[m](torch.cat([local[m],common],dim=-1))
            candidate=(local[m]+torch.sigmoid(supplement)*(common-local[m]))*mask[...,m,None]
            enhanced.append(candidate)
            scores.append(self.trust[m](torch.cat([candidate,common],dim=-1)))
        score=torch.cat(scores,dim=-1).masked_fill(~mask,-1e4)
        weight=score.softmax(-1)*mask.float()
        weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-8)
        mixture=sum(weight[...,m,None]*enhanced[m] for m in range(3))
        fused=self.combine(torch.cat([mixture,common],dim=-1))
        return self.finish(fused,enhanced,batch)


VARIANTS={"text_anchor":TextAnchor,"shared_private":SharedPrivate,
          "reliability_proxy":ReliabilityProxy,"dual_query":DualQueryFusion,
          "enhance_balance":EnhanceBalance}
