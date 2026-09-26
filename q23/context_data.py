"""Contest-aligned BERT contextual features with leakage-free masked variants."""
from __future__ import annotations

import numpy as np
import torch

from .core import MODEL_ID, MODEL_REVISION, OUT, pack_split


def encode_with_model(model,ids,attention,token_types=None,device="cuda",batch_size=128):
    inputs = torch.as_tensor(ids, dtype=torch.long)
    attn = torch.as_tensor(attention, dtype=torch.long)
    types = torch.zeros_like(inputs) if token_types is None else torch.as_tensor(token_types, dtype=torch.long)
    rows = []
    with torch.inference_mode():
        for i in range(0, len(inputs), batch_size):
            result = model(input_ids=inputs[i:i+batch_size].to(device),
                           attention_mask=attn[i:i+batch_size].to(device),
                           token_type_ids=types[i:i+batch_size].to(device)).last_hidden_state
            rows.append(result.cpu().to(torch.float16).numpy())
    return np.concatenate(rows)


def encode_bert(ids, attention, token_types=None, device="cuda", batch_size=128):
    from transformers import BertModel
    model = BertModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
    result = encode_with_model(model,ids,attention,token_types,device,batch_size)
    del model
    return result


def attach_text(split, packed, device, stats=None):
    if "text" in split:
        packed["text"] = torch.as_tensor(np.asarray(split["text"], dtype=np.float16), device=device).float()
    else:
        ids = np.asarray(split["text_bert"])[:,0,:]
        attn = np.asarray(split["text_bert"])[:,1,:]
        types = np.asarray(split["text_bert"])[:,2,:]
        packed["text"] = torch.as_tensor(encode_bert(ids,attn,types,device),device=device).float()
    packed["text"] *= packed["mask"][...,0,None]
    return packed


def masked_training_cache(split, device):
    cache_dir = OUT / "experiment_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "masked_text_2026.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            return (torch.as_tensor(z["text"], device=device).float(),
                    torch.as_tensor(z["mask"], device=device))
    ids = np.asarray(split["text_bert"])[:,0,:].astype(np.int64).copy()
    attention = np.asarray(split["text_bert"])[:,1,:].astype(np.int64).copy()
    types = np.asarray(split["text_bert"])[:,2,:].astype(np.int64)
    mask = (attention > 0) & (ids != 101) & (ids != 102) & (ids != 0)
    rng = np.random.default_rng(2026)
    for i in range(len(ids)):
        pos = np.flatnonzero(mask[i])
        if len(pos) < 2:
            continue
        size = max(1,round(len(pos)*rng.uniform(.15,.5)))
        start = int(rng.integers(0,len(pos)-size+1))
        masked = pos[start:start+size]
        ids[i,masked] = 0
        attention[i,masked] = 0
        mask[i,masked] = False
    encoded = encode_bert(ids,attention,types,device)
    encoded *= mask[...,None]
    np.savez(path, text=encoded, mask=mask)
    return torch.as_tensor(encoded,device=device).float(), torch.as_tensor(mask,device=device)


def augment(batch, masked_text, masked_mask, source_indices, rng):
    out = {k:v.clone() for k,v in batch.items()}
    for row, source in enumerate(source_indices.tolist()):
        if rng.random() < .2:
            continue
        if rng.random() < .35:
            out["text"][row] = masked_text[source]
            out["mask"][row,:,0] = masked_mask[source]
        modes = ((1,), (2,), (1,2), ())[int(rng.choice(4, p=[.3,.3,.3,.1]))]
        for m in modes:
            pos = torch.nonzero(out["mask"][row,:,m],as_tuple=False).flatten().cpu().numpy()
            if len(pos) < 2:
                continue
            size = max(1,round(len(pos)*rng.uniform(.1,.5)))
            start = int(rng.integers(0,len(pos)-size+1))
            out["mask"][row,pos[start:start+size],m] = False
        out["audio"][row] *= out["mask"][row,:,1,None]
        out["vision"][row] *= out["mask"][row,:,2,None]
    return out
