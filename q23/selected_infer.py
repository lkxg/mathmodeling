"""Final attachment inference with the validation-selected contextual model."""
from __future__ import annotations

import json

import numpy as np
import torch

from .context_data import attach_text
from .core import DATA, OUT, pack_split, read_pkl, save_csv, save_json, take
from .infer import LABELS, NAMES, evidence_windows, explain_one
from .selected_runtime import load_selected


def predict_batches(model,data,batch_size=128):
    logits=[]
    regs=[]
    model.eval()
    with torch.inference_mode():
        for index in torch.arange(len(data["ids"]),device=data["ids"].device).split(batch_size):
            out=model(take(data,index))
            logits.append(out["logits"].cpu())
            regs.append(out["reg"].cpu())
    return torch.cat(logits).numpy(),torch.cat(regs).numpy()


def stack_q2(folder):
    paths=sorted(folder.glob("*.pkl"))
    records=[read_pkl(p)["test"] for p in paths]
    combined={k:np.concatenate([np.asarray(x[k]) for x in records],axis=0)
              for k in ("text_bert","audio","vision")}
    return paths,combined


def stack_q3(folder):
    paths=sorted(folder.glob("*.pkl"))
    records=[read_pkl(p) for p in paths]
    combined={k:np.stack([np.asarray(x[k]) for x in records],axis=0)
              for k in ("text_bert","text","audio","vision")}
    return paths,records,combined


def coherent(label,reg):
    return 0. if label==1 else min(reg,-1e-6) if label==0 else max(reg,1e-6)


def main():
    torch.set_num_threads(8)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model,choice=load_selected(device)
    with np.load(OUT/"normalization.npz") as z:
        stats={k:z[k] for k in z.files}
    folder=DATA/"附件3-模态缺失特征样本/对齐版本"
    paths,raw=stack_q2(folder)
    data,_=pack_split(raw,device,stats)
    attach_text(raw,data,device)
    logits,reg=predict_batches(model,data)
    probs=torch.softmax(torch.as_tensor(logits),-1).numpy()
    rows=[]
    for i,p in enumerate(paths):
        label=int(logits[i].argmax())
        rows.append({"sample_id":p.stem,"polarity":LABELS[label],"class_id":label,
                     "intensity":round(coherent(label,float(reg[i])),6),"raw_intensity":round(float(reg[i]),6),
                     "prob_negative":round(float(probs[i,0]),6),
                     "prob_neutral":round(float(probs[i,1]),6),
                     "prob_positive":round(float(probs[i,2]),6)})
    save_csv(OUT/"experiments"/"attachment3_predictions_selected.csv",rows)

    folder=DATA/"附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"
    paths,records,raw=stack_q3(folder)
    data,_=pack_split(raw,device,stats)
    attach_text(raw,data,device)
    summary=[]
    details=[]
    audit=[]
    for i,p in enumerate(paths):
        item=take(data,slice(i,i+1))
        exp=explain_one(model,item)
        windows=evidence_windows(model,item,exp["predicted_class"])
        exp["reported_intensity"]=coherent(exp["predicted_class"],exp["intensity"])
        exp.update({"sample_id":p.stem,"raw_text":str(records[i]["raw_text"]),
                    "token_ids":item["ids"][0].cpu().tolist(),"windows":windows})
        details.append(exp)
        avail=item["mask"][0].any(0).cpu().tolist()
        audit.append({"sample_id":p.stem,"nonzero_positions":{
            m:int(item["mask"][0,:,j].sum().item()) for j,m in enumerate(NAMES)},
            "modality_available":dict(zip(NAMES,avail))})
        summary.append({"sample_id":p.stem,"polarity":exp["predicted_label"],
                        "class_id":exp["predicted_class"],"intensity":round(exp["reported_intensity"],6),
                        "raw_intensity":round(exp["intensity"],6),
                        "main_modality":exp["main_modality"],"vision_available":avail[2],
                        **{f"contribution_{m}":round(exp["modality_contribution"][m],6) for m in NAMES},
                        **{f"share_{m}":round(exp["modality_share"][m],6) for m in NAMES},
                        "evidence_positions":";".join(f"{w['modality']}:{w['start_position']}-{w['end_position_exclusive']-1}" for w in windows)})
    save_csv(OUT/"experiments"/"attachment4_predictions_explanations_selected.csv",summary)
    save_json(OUT/"experiments"/"attachment4_explanations_selected.json",details)
    save_json(OUT/"experiments"/"attachment4_input_audit_selected.json",audit)
    print("selected",choice["selected_models"],"attachment3",len(rows),"attachment4",len(summary),flush=True)


if __name__=="__main__":
    main()
