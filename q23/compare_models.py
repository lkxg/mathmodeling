"""Train and compare three paper-inspired architectures on one fixed split."""
from __future__ import annotations

import argparse
import gc
import time

import numpy as np
import torch
from torch.nn import functional as F

from .context_data import attach_text, augment, masked_training_cache
from .core import DATA, OUT, metrics, pack_split, read_pkl, save_csv, save_json, seed_all, take
from .fusion_variants import VARIANTS


@torch.no_grad()
def predict_variant(model, data, batch_size=256):
    model.eval()
    logits, reg = [], []
    for indices in torch.arange(len(data["ids"]),device=data["ids"].device).split(batch_size):
        result = model(take(data,indices))
        logits.append(result["logits"].cpu())
        reg.append(result["reg"].cpu())
    return torch.cat(logits).numpy(), torch.cat(reg).numpy()


def loss_variant(result,batch):
    loss=F.cross_entropy(result["logits"],batch["class"])+.35*F.huber_loss(result["reg"],batch["reg"])
    for m in range(3):
        ix=result["availability"][:,m]
        if ix.any():
            loss=loss+.06*F.cross_entropy(result["aux"][ix,m,:3],batch["class"][ix])
    return loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=25)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--models",nargs="+",choices=list(VARIANTS),
                        default=["text_anchor","shared_private","reliability_proxy"])
    args=parser.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(8)
    raw=read_pkl(DATA/"附件2-数据集特征文件/aligned_50.pkl")
    train,stats=pack_split(raw["train"],device)
    valid,_=pack_split(raw["valid"],device,stats)
    attach_text(raw["train"],train,device)
    attach_text(raw["valid"],valid,device)
    masked_text,masked_mask=masked_training_cache(raw["train"],device)
    results=[]
    OUT.joinpath("experiments").mkdir(parents=True,exist_ok=True)
    for name in args.models:
        seed_all(2026)
        rng=np.random.default_rng(2026)
        model=VARIANTS[name]().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=.03)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=2e-5)
        best=-1e9
        best_epoch=0
        best_state=None
        history=[]
        started=time.time()
        for epoch in range(1,args.epochs+1):
            model.train()
            perm=torch.randperm(len(train["ids"]),device=device)
            losses=[]
            for indices in perm.split(args.batch_size):
                batch=augment(take(train,indices),masked_text,masked_mask,indices,rng)
                opt.zero_grad(set_to_none=True)
                result=model(batch)
                loss=loss_variant(result,batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            logits,reg=predict_variant(model,valid)
            measured=metrics(logits,reg,valid)
            score=measured["accuracy"]+.05*measured["macro_f1"]-.02*measured["mae"]
            history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),**measured})
            if score>best:
                best=score
                best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch==1 or epoch%5==0:
                print(f"{name} epoch={epoch} accuracy={measured['accuracy']:.4f} f1={measured['macro_f1']:.4f} mae={measured['mae']:.4f}",flush=True)
        ckpt=OUT/"experiments"/f"{name}.pt"
        torch.save({"name":name,"state_dict":best_state,"epoch":best_epoch,
                    "selection":"accuracy + 0.05*macro_f1 - 0.02*MAE",
                    "seed":2026},ckpt)
        model.load_state_dict(best_state)
        logits,reg=predict_variant(model,valid)
        summary={"model":name,"best_epoch":best_epoch,"seconds":time.time()-started,
                 **metrics(logits,reg,valid)}
        results.append(summary)
        save_json(OUT/"experiments"/f"{name}_history.json",history)
        np.savez(OUT/"experiments"/f"{name}_valid_predictions.npz",logits=logits,reg=reg)
        print("selected",summary,flush=True)
        del model,opt,scheduler
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    comparison_name=("comparison.csv" if args.models==["text_anchor","shared_private","reliability_proxy"]
                     else "comparison_"+"_".join(args.models)+".csv")
    save_csv(OUT/"experiments"/comparison_name,results)
    print("all models complete",flush=True)


if __name__=="__main__":
    main()
