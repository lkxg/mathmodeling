"""Small native-sequence encoders with separately auditable attention weights."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn

STREAMS = ("text", "emotion", "acoustic", "vision")
DIMENSIONS = {"text":768, "emotion":768, "acoustic":25, "vision":28}
MODES = ("hard_overlap", "time_kernel", "local_attention", "local_no_penalty", "global_attention")


def fit_normalizers(samples, train):
    """Streaming frame statistics, fitted only on the specified training clips."""
    result = {}
    for m in STREAMS:
        count = 0; total = np.zeros(DIMENSIONS[m]); squares = total.copy()
        for i in train:
            source = samples[i][m]
            mask = source["content_valid"] if m == "text" else source["quality"] > 0
            x = source["x"][mask].astype(np.float64)
            count += len(x); total += x.sum(axis=0); squares += (x*x).sum(axis=0)
        mean = total / max(count, 1)
        scale = np.sqrt(np.maximum(0, squares/max(count, 1)-mean*mean))
        scale[scale < 1e-6] = 1.
        result[m] = {"mean":mean.astype(np.float32), "scale":scale.astype(np.float32), "frames":count}
    return result


def prepare_batch(samples, indices, normalizers, device, step=.5):
    selected = [samples[i] for i in indices]
    anchors = []
    for s in selected:
        edges = np.r_[np.arange(0, s["media"]["duration"], step), s["media"]["duration"]]
        anchors.append(np.column_stack([edges[:-1], edges[1:]]))
    b, length = len(selected), max(map(len, anchors))
    at = np.zeros((b,length,2), np.float32); am = np.zeros((b,length), bool)
    for i,a in enumerate(anchors): at[i,:len(a)] = a; am[i,:len(a)] = True
    result = {"anchor_times":torch.as_tensor(at, device=device),
              "anchor_mask":torch.as_tensor(am, device=device)}
    for m in STREAMS:
        n = max(1, max(len(s[m]["x"]) for s in selected))
        x = np.zeros((b,n,DIMENSIONS[m]),np.float32)
        times = np.zeros((b,n,2),np.float32)
        quality = np.zeros((b,n),np.float32); content = np.zeros((b,n),bool)
        for i,s in enumerate(selected):
            src = s[m]; nnative = len(src["x"])
            good = src["content_valid"] if m == "text" else src["quality"] > 0
            xx = (src["x"].astype(np.float32)-normalizers[m]["mean"])/normalizers[m]["scale"]
            xx[~good] = 0
            if not np.isfinite(xx).all(): raise ValueError("Nonfinite normalized native features")
            x[i,:nnative] = xx; times[i,:nnative] = src["intervals"]
            quality[i,:nnative] = src["quality"]; content[i,:nnative] = good
        overlap = np.maximum(0, np.minimum(at[:,:,None,1],times[:,None,:,1])-np.maximum(at[:,:,None,0],times[:,None,:,0]))
        distance = times.mean(axis=-1)[:,None,:]-at.mean(axis=-1)[:,:,None]
        result[m] = {k:torch.as_tensor(v, device=device) for k,v in
                     {"x":x,"quality":quality,"content":content,"times":times,"overlap":overlap,"distance":distance}.items()}
    return result


def attention_weights(mode, logits, distance, overlap, quality, anchor_mask, window=.5, tau=.25):
    valid = (quality[:,None,:] > 0) & anchor_mask[:,:,None]
    if mode == "hard_overlap":
        unnormalized = overlap * quality[:,None,:] * valid
        return unnormalized / unnormalized.sum(-1,keepdim=True).clamp_min(1e-12)
    if mode not in MODES: raise ValueError(mode)
    if mode != "global_attention": valid = valid & (distance.abs() <= window)
    score = torch.zeros_like(logits) if mode == "time_kernel" else logits
    if mode in ("time_kernel", "local_attention"): score = score - .5*(distance/tau)**2
    score = score + quality[:,None,:].clamp_min(1e-12).log()
    weights = torch.softmax(score.masked_fill(~valid, -1e9), dim=-1) * valid
    return weights / weights.sum(-1,keepdim=True).clamp_min(1e-12)


def masked_moments(x, mask):
    weights = mask.to(x.dtype); denom = weights.sum(1,keepdim=True).clamp_min(1.)
    mean = (x*weights[:,:,None]).sum(1)/denom
    variance = ((x-mean[:,None,:]).square()*weights[:,:,None]).sum(1)/denom
    # Smooth derivative at zero; missing streams stay exactly zero.
    std = (torch.sqrt(variance+1e-8)-1e-4) * mask.any(1)[:,None]
    return mean, std


class NativeAlignment(nn.Module):
    def __init__(self, mode, latent=16, window=.5, tau=.25):
        super().__init__()
        self.mode, self.window, self.tau, self.latent = mode, window, tau, latent
        self.encoders = nn.ModuleDict({m:nn.Sequential(nn.Linear(d,latent),nn.LayerNorm(latent),nn.GELU()) for m,d in DIMENSIONS.items()})
        self.query_token = nn.Parameter(torch.zeros(latent))
        self.query = nn.Linear(latent,latent,bias=False)
        self.keys = nn.ModuleDict({m:nn.Linear(latent,latent,bias=False) for m in STREAMS[1:]})
        self.head = nn.Sequential(nn.Linear(latent*7+4,32),nn.GELU(),nn.Dropout(.1))
        self.regression = nn.Linear(32,1); self.classification = nn.Linear(32,3)

    def forward(self, batch, return_attention=False):
        am = batch["anchor_mask"]; text = batch["text"]
        tx = self.encoders["text"](text["x"])
        tmean,_ = masked_moments(tx,text["content"])
        tw = attention_weights("hard_overlap",text["overlap"],text["distance"],text["overlap"],text["quality"],am)
        query = self.query(tw@tx+self.query_token)
        parts = [tmean]; flags = [text["content"].any(1).to(tx.dtype)]
        maps = {}
        for m in STREAMS[1:]:
            src = batch[m]; latent = self.encoders[m](src["x"])
            logits = query@self.keys[m](latent).transpose(1,2)/math.sqrt(self.latent)
            w = attention_weights(self.mode,logits,src["distance"],src["overlap"],src["quality"],am,self.window,self.tau)
            observed = (w.sum(-1)>0)&am
            mean,std = masked_moments(w@latent,observed)
            parts.extend([mean,std]); flags.append(observed.sum(1)/am.sum(1).clamp_min(1))
            if return_attention: maps[m] = w
        embedding = self.head(torch.cat(parts+[torch.stack(flags,dim=1)],dim=1))
        result = (self.regression(embedding).squeeze(-1), self.classification(embedding))
        return (*result,maps) if return_attention else result
