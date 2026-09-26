"""Compare individual models and equal-weight ensembles using validation only."""
from __future__ import annotations

import itertools

import numpy as np

from .core import DATA, OUT, metrics, pack_split, read_pkl, save_csv, save_json
from .fusion_variants import VARIANTS


def main():
    raw=read_pkl(DATA/"附件2-数据集特征文件/aligned_50.pkl")
    train,stats=pack_split(raw["train"],"cpu")
    valid,_=pack_split(raw["valid"],"cpu",stats)
    names=[n for n in ("text_anchor","shared_private","reliability_proxy")
           if (OUT/"experiments"/f"{n}.pt").exists()]
    if len(names)!=3:
        raise RuntimeError("Train all three variants before selection")
    loaded={}
    for n in names:
        with np.load(OUT/"experiments"/f"{n}_valid_predictions.npz") as z:
            loaded[n]=(z["logits"],z["reg"])
    rows=[]
    for count in (1,2,3):
        for combo in itertools.combinations(names,count):
            logits=np.mean([loaded[n][0] for n in combo],axis=0)
            reg=np.mean([loaded[n][1] for n in combo],axis=0)
            result=metrics(logits,reg,valid)
            rows.append({"models":"+".join(combo),"count":count,**result})
    rows.sort(key=lambda r:(r["accuracy"],r["macro_f1"],-r["mae"]),reverse=True)
    save_csv(OUT/"experiments"/"selection.csv",rows)
    chosen=rows[0]
    save_json(OUT/"experiments"/"selection.json",{
        "selected_models":chosen["models"].split("+"),
        "rule":"max validation Accuracy, then macro-F1, then lower MAE",
        "validation":chosen,
        "all_candidates":rows})
    for r in rows:print(r["models"],f"Acc={r['accuracy']:.4f}",f"F1={r['macro_f1']:.4f}",f"MAE={r['mae']:.4f}")


if __name__=="__main__":
    main()
