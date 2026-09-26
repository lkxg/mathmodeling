"""Train a query-fusion student with complete-input teacher distillation.

The correlation loss borrows the broad idea of CMAD; this is a small,
task-specific adaptation rather than a reproduction of its full method.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.nn import functional as F

from .compare_models import loss_variant,predict_variant
from .context_data import attach_text,augment,masked_training_cache
from .core import DATA,OUT,metrics,pack_split,read_pkl,save_json,seed_all,take
from .fusion_variants import VARIANTS
from .selected_runtime import load_selected


@torch.inference_mode()
def cache_teacher(teacher,train):
    outputs=[]
    teacher.eval()
    for index in torch.arange(len(train["ids"]),device=train["ids"].device).split(256):
        outputs.append(teacher(take(train,index)))
    return {key:torch.cat([out[key] for out in outputs]).detach()
            for key in ("logits","reg","representation")}


def loss_with_distillation(student,batch,teacher,weight):
    supervised=loss_variant(student,batch)
    temperature=2.
    kl=F.kl_div(F.log_softmax(student["logits"]/temperature,dim=-1),
                F.softmax(teacher["logits"]/temperature,dim=-1),
                reduction="batchmean")*temperature*temperature
    regression=F.huber_loss(student["reg"],teacher["reg"])
    student_repr=F.normalize(student["representation"],dim=-1)
    teacher_repr=F.normalize(teacher["representation"],dim=-1)
    correlation=F.mse_loss(student_repr@student_repr.T,teacher_repr@teacher_repr.T)
    return supervised+weight*(.25*kl+.1*regression+.05*correlation)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--student",choices=["dual_query","enhance_balance"],default="dual_query")
    parser.add_argument("--epochs",type=int,default=25)
    args=parser.parse_args()
    seed_all(2026)
    torch.set_num_threads(8)
    device="cuda" if torch.cuda.is_available() else "cpu"
    raw=read_pkl(DATA/"附件2-数据集特征文件/aligned_50.pkl")
    train,stats=pack_split(raw["train"],device)
    valid,_=pack_split(raw["valid"],device,stats)
    attach_text(raw["train"],train,device)
    attach_text(raw["valid"],valid,device)
    masked_text,masked_mask=masked_training_cache(raw["train"],device)
    teacher,_=load_selected(device)
    teacher_outputs=cache_teacher(teacher,train)
    del teacher
    model=VARIANTS[args.student]().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=4e-4,weight_decay=.03)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=2e-5)
    rng=np.random.default_rng(2026)
    best=-1e9
    best_state=None
    best_epoch=0
    history=[]
    started=time.time()
    for epoch in range(1,args.epochs+1):
        model.train()
        order=torch.randperm(len(train["ids"]),device=device)
        losses=[]
        for index in order.split(128):
            batch=augment(take(train,index),masked_text,masked_mask,index,rng)
            targets={k:v[index] for k,v in teacher_outputs.items()}
            opt.zero_grad(set_to_none=True)
            result=model(batch)
            weight=min(1.,epoch/5)
            loss=loss_with_distillation(result,batch,targets,weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        logits,reg=predict_variant(model,valid)
        result=metrics(logits,reg,valid)
        score=result["accuracy"]+.05*result["macro_f1"]-.02*result["mae"]
        history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),**result})
        if score>best:
            best=score
            best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%5==0:
            print(f"{args.student} distilled epoch={epoch} acc={result['accuracy']:.4f} f1={result['macro_f1']:.4f} mae={result['mae']:.4f}",flush=True)
    name=f"{args.student}_distilled"
    root=OUT/"experiments"
    torch.save({"name":args.student,"state_dict":best_state,"epoch":best_epoch,
                "teacher":["shared_private","reliability_proxy"],
                "distillation":"KL+regression+relation", "seed":2026},root/f"{name}.pt")
    model.load_state_dict(best_state)
    logits,reg=predict_variant(model,valid)
    np.savez(root/f"{name}_valid_predictions.npz",logits=logits,reg=reg)
    save_json(root/f"{name}_history.json",history)
    save_json(root/f"{name}_summary.json",{"best_epoch":best_epoch,"seconds":time.time()-started,
                                          **metrics(logits,reg,valid)})
    print("selected",name,best_epoch,metrics(logits,reg,valid),flush=True)


if __name__=="__main__":
    main()
