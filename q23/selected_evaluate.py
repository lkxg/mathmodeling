"""Validate selected model under local gaps; evaluate held-out test once."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .context_data import attach_text, encode_with_model
from .core import DATA, MODEL_ID, MODEL_REVISION, OUT, metrics, pack_split, read_pkl, save_csv, save_json
from .selected_infer import predict_batches
from .selected_runtime import load_selected


def mask_scenario(data, raw_text_bert, modes, rate, location, bert=None, device="cuda"):
    changed={k:v.clone() for k,v in data.items()}
    mask=changed["mask"]
    ids=np.asarray(raw_text_bert)[:,0,:].astype(np.int64).copy()
    attention=np.asarray(raw_text_bert)[:,1,:].astype(np.int64).copy()
    types=np.asarray(raw_text_bert)[:,2,:].astype(np.int64)
    for i in range(len(mask)):
        pos=torch.nonzero(changed["valid"][i],as_tuple=False).flatten().cpu().numpy()
        if not len(pos):continue
        length=max(1,round(len(pos)*rate))
        start=0 if location=="start" else (len(pos)-length)//2 if location=="middle" else len(pos)-length
        selected=pos[start:start+length]
        for mode in modes:
            mask[i,selected,mode]=False
        if 0 in modes:
            ids[i,selected]=0
            attention[i,selected]=0
    changed["audio"]*=mask[...,1,None]
    changed["vision"]*=mask[...,2,None]
    if 0 in modes:
        if bert is None:raise RuntimeError("BERT model required for text-missing evaluation")
        contextual=encode_with_model(bert,ids,attention,types,device)
        changed["text"]=torch.as_tensor(contextual,device=device).float()*mask[...,0,None]
    return changed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--include-test",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(8)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,choice=load_selected(device)
    with np.load(OUT/"normalization.npz") as z:
        stats={k:z[k] for k in z.files}
    raw=read_pkl(DATA/"附件2-数据集特征文件/aligned_50.pkl")
    valid,_=pack_split(raw["valid"],device,stats)
    attach_text(raw["valid"],valid,device)
    logits,reg=predict_batches(model,valid)
    complete=metrics(logits,reg,valid)
    rows=[{"scenario":"complete","modality":"all","rate":0,"location":"none",**complete}]
    from transformers import BertModel
    bert=BertModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION).to(device).eval()
    for mode,name in enumerate(("text","audio","vision")):
        for rate in (.1,.3,.5):
            for location in ("start","middle","end"):
                changed=mask_scenario(valid,raw["valid"]["text_bert"],(mode,),rate,location,bert,device)
                measured=metrics(*predict_batches(model,changed),valid)
                rows.append({"scenario":"single_contiguous","modality":name,
                             "rate":rate,"location":location,**measured})
    for modes in ((0,1),(0,2),(1,2),(0,1,2)):
        changed=mask_scenario(valid,raw["valid"]["text_bert"],modes,.3,"middle",bert,device)
        measured=metrics(*predict_batches(model,changed),valid)
        rows.append({"scenario":"multiple_contiguous","modality":"+".join(("text","audio","vision")[j] for j in modes),
                     "rate":.3,"location":"middle",**measured})
    del bert
    save_csv(OUT/"experiments"/"selected_robustness.csv",rows)
    report={"selected_models":choice["selected_models"],"validation_complete":complete,
            "validation_single_mean_accuracy":float(np.mean([r["accuracy"] for r in rows if r["scenario"]=="single_contiguous"])),
            "validation_single_mean_macro_f1":float(np.mean([r["macro_f1"] for r in rows if r["scenario"]=="single_contiguous"])),
            "validation_multiple_mean_accuracy":float(np.mean([r["accuracy"] for r in rows if r["scenario"]=="multiple_contiguous"])),
            "validation_multiple_mean_macro_f1":float(np.mean([r["macro_f1"] for r in rows if r["scenario"]=="multiple_contiguous"]))}
    if args.include_test:
        test,_=pack_split(raw["test"],device,stats)
        attach_text(raw["test"],test,device)
        report["held_out_test_complete"]=metrics(*predict_batches(model,test),test)
    save_json(OUT/"experiments"/"selected_metrics.json",report)
    print(report,flush=True)


if __name__=="__main__":
    main()
