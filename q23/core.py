from __future__ import annotations

import csv
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "E题/E题数据"
OUT = ROOT / "outputs/q23"
MODEL_ID = "google-bert/bert-base-uncased"
MODEL_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_pkl(path: Path):
    # Pickle files are supplied with the contest; never load untrusted files here.
    with path.open("rb") as f:
        return pickle.load(f)


def load_embedding(device):
    from transformers import BertModel
    model = BertModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval()
    weight = model.embeddings.word_embeddings.weight.detach().clone()
    del model
    return nn.Embedding.from_pretrained(weight.to(device), freeze=True, padding_idx=0)


def pack_split(split, device, stats=None):
    ids = np.asarray(split["text_bert"])[:, 0, :].astype(np.int64)
    attn = np.asarray(split["text_bert"])[:, 1, :] > 0
    aud = np.asarray(split["audio"], dtype=np.float32)
    vis = np.asarray(split["vision"], dtype=np.float32)
    text_mask = attn & (ids != 0) & (ids != 101) & (ids != 102)
    # A boundary token is a sequence position only if a real A/V observation exists there.
    valid = text_mask | np.any(aud != 0, axis=-1) | np.any(vis != 0, axis=-1)
    amask = np.any(aud != 0, axis=-1) & valid
    vmask = np.any(vis != 0, axis=-1) & valid
    if stats is None:
        means, stds = [], []
        for x, mask in ((aud, amask), (vis, vmask)):
            observed = x[mask]
            means.append(observed.mean(axis=0))
            stds.append(np.maximum(observed.std(axis=0), 1e-5))
        stats = {"audio_mean": means[0], "audio_std": stds[0],
                 "vision_mean": means[1], "vision_std": stds[1]}
    aud = np.where(amask[..., None], (aud - stats["audio_mean"]) / stats["audio_std"], 0)
    vis = np.where(vmask[..., None], (vis - stats["vision_mean"]) / stats["vision_std"], 0)
    aud = np.clip(aud, -8, 8)
    vis = np.clip(vis, -8, 8)
    item = {"ids": torch.as_tensor(ids, device=device),
            "audio": torch.as_tensor(aud, device=device),
            "vision": torch.as_tensor(vis, device=device),
            "mask": torch.as_tensor(np.stack([text_mask, amask, vmask], axis=-1), device=device),
            "valid": torch.as_tensor(valid, device=device)}
    if "classification_labels" in split:
        item["class"] = torch.as_tensor(np.asarray(split["classification_labels"], dtype=np.int64), device=device)
        item["reg"] = torch.as_tensor(np.asarray(split["regression_labels"], dtype=np.float32), device=device)
    return item, stats


def take(data, idx):
    return {k: v[idx] for k, v in data.items()}


def block_mask(batch, rng=None):
    """Apply contiguous intra-modal gaps only to currently observed positions."""
    rng = rng or np.random.default_rng()
    out = {k: v.clone() for k, v in batch.items()}
    masks = out["mask"]
    b, length, _ = masks.shape
    for n in range(b):
        if rng.random() < .2:
            continue
        modes = rng.choice(3, size=int(rng.choice([1, 1, 1, 2, 2, 3])), replace=False)
        valid_pos = torch.nonzero(out["valid"][n], as_tuple=False).flatten().cpu().numpy()
        if not len(valid_pos):
            continue
        lo, hi = int(valid_pos.min()), int(valid_pos.max()) + 1
        span = max(1, round((hi-lo) * float(rng.uniform(.1, .6))))
        start = int(rng.integers(lo, max(lo + 1, hi - span + 1)))
        masks[n, start:start+span, modes] = False
    out["ids"] = torch.where(masks[..., 0], out["ids"], 0)
    out["audio"] = out["audio"] * masks[..., 1, None]
    out["vision"] = out["vision"] * masks[..., 2, None]
    return out


class Predictor(nn.Module):
    def __init__(self, embedding, hidden=96, dropout=.2):
        super().__init__()
        self.embedding = embedding
        self.project = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU())
                                      for d in (768, 74, 35)])
        self.position = nn.Embedding(50, hidden)
        self.gate = nn.Sequential(nn.Linear(hidden * 3 + 3, hidden), nn.GELU(), nn.Linear(hidden, 3))
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden*2, dropout, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.pool = nn.Linear(hidden, 1)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4)) for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 4))

    def forward(self, batch):
        mask, valid = batch["mask"], batch["valid"]
        n, l = valid.shape
        x = [self.project[0](self.embedding(batch["ids"])),
             self.project[1](batch["audio"]), self.project[2](batch["vision"])]
        x = [h * mask[..., m, None] for m, h in enumerate(x)]
        scores = self.gate(torch.cat(x + [mask.float()], dim=-1)).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1) * mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        fused = sum(x[m] * weights[..., m, None] for m in range(3))
        fused = fused + self.position(torch.arange(l, device=valid.device))[None]
        fused = self.temporal(fused, src_key_padding_mask=~valid)
        pool_weight = self.pool(fused).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)
        pooled = (fused * pool_weight[..., None]).sum(dim=1)
        summaries = []
        for m in range(3):
            denom = mask[..., m].sum(1, keepdim=True).clamp_min(1)
            summaries.append(x[m].sum(1) / denom)
        availability = mask.any(1)
        share = mask.float().sum(1)
        share = share / share.sum(1, keepdim=True).clamp_min(1)
        expert = sum(self.experts[m](summaries[m]) * share[:, m, None] for m in range(3))
        main = self.head(torch.cat([pooled, sum(summaries)/3], dim=-1))
        logits = main[:, :3] + .3 * expert[:, :3]
        reg = 3 * torch.tanh((main[:, 3] + .3 * expert[:, 3]) / 3)
        aux = torch.stack([self.experts[m](summaries[m]) for m in range(3)], dim=1)
        return {"logits": logits, "reg": reg, "aux": aux,
                "availability": availability, "gate": weights, "pool": pool_weight}


def loss_fn(pred, batch):
    ce = F.cross_entropy(pred["logits"], batch["class"])
    huber = F.huber_loss(pred["reg"], batch["reg"])
    aux_loss = 0
    for m in range(3):
        ix = pred["availability"][:, m]
        if ix.any():
            aux_loss = aux_loss + F.cross_entropy(pred["aux"][ix, m, :3], batch["class"][ix])
    return ce + .5 * huber + .1 * aux_loss


@torch.no_grad()
def predict(model, data, batch_size=256):
    model.eval()
    logits, reg = [], []
    for idx in torch.arange(len(data["ids"]), device=data["ids"].device).split(batch_size):
        out = model(take(data, idx))
        logits.append(out["logits"].cpu())
        reg.append(out["reg"].cpu())
    return torch.cat(logits).numpy(), torch.cat(reg).numpy()


def metrics(logits, reg, data):
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
    y = data["class"].cpu().numpy()
    r = data["reg"].cpu().numpy()
    pred = logits.argmax(-1)
    corr = float(np.corrcoef(r, reg)[0, 1]) if np.std(reg) > 1e-8 else 0.
    coherent = np.where(pred == 1, 0., np.where(pred == 0, np.minimum(reg, -1e-6), np.maximum(reg, 1e-6)))
    coherent_corr = float(np.corrcoef(r, coherent)[0, 1]) if np.std(coherent) > 1e-8 else 0.
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "mae": float(mean_absolute_error(r, reg)), "pearson": corr,
            "mae_coherent_output": float(mean_absolute_error(r, coherent)),
            "pearson_coherent_output": coherent_corr}


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x), encoding="utf-8")
