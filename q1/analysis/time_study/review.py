"""Review all 18 flagged cases; human annotations remain explicitly separate."""
from __future__ import annotations
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
import soundfile as sf

from ...common import atomic_json, atomic_npz
from ..alignment_quality import speech_probability, grid_masks, WINDOW
from ..probe import clean_json
from . import data
from .data import default_config, load_samples, output_dir, provenance


def review(cfg=None):
    import cv2
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from silero_vad import load_silero_vad
    torch.set_num_threads(2); cv2.setNumThreads(1); plt.rcParams["text.parse_math"]=False
    cfg=cfg or default_config(); out=output_dir(cfg,"review")
    vad=pd.read_csv(Path(cfg["output_dir"])/"analysis"/"alignment_vad.csv")
    selected=vad[vad.needs_review].set_index("sample_id")
    samples=[s for s in load_samples(cfg) if s["id"] in selected.index]
    model=load_silero_vad()
    notes_path=Path(__file__).with_name("scene_review.json")
    notes={r["sample_id"]:r for r in json.loads(notes_path.read_text())["cases"]} if notes_path.exists() else {}
    cases=[]; word_rows=[]; contacts=[]
    for s in samples:
        wave,sr=sf.read(s["directory"]/"audio.wav",dtype="float32")
        probs=speech_probability(model,wave)
        wordmask,speech,_=grid_masks(s["words"],probs,s["media"])
        row=selected.loc[s["id"]]
        per_word=[]
        for wi,w in enumerate(s["words"]):
            a,b=w["start"],w["end"]
            start=int(max(0,(a-s["media"]["audio_offset"])*sr))
            end=int(min(len(wave),(b-s["media"]["audio_offset"])*sr))
            speech_fraction=None
            if w["time_valid"] and end>start:
                centers=(np.arange(len(speech))+.5)*.01
                keep=(centers>=a)&(centers<b)
                if keep.any(): speech_fraction=float(speech[keep].mean())
            wr={"sample_id":s["id"],"word_index":wi,"word":w["text"],
                "auto_start":a,"auto_end":b,"auto_valid":w["time_valid"],
                "flags":";".join(w["flags"]),"vad_speech_fraction":speech_fraction,
                "manual_start":"","manual_end":"","reviewer_code":"","comment":""}
            word_rows.append(wr); per_word.append(wr)
        fig,ax=plt.subplots(3,1,figsize=(12,5),sharex=True,layout="constrained")
        times=np.arange(len(wave))/sr+s["media"]["audio_offset"]
        stride=max(1,len(wave)//12000)
        ax[0].plot(times[::stride],wave[::stride],lw=.45); ax[0].set_ylabel("Wave")
        pt=(np.arange(len(probs))+.5)*WINDOW/sr+s["media"]["audio_offset"]
        ax[1].plot(pt,probs,lw=.7,label="VAD probability"); ax[1].axhline(.5,color="grey",ls="--")
        ax[1].set_ylabel("VAD"); ax[1].set_ylim(-.02,1.02)
        for w in s["words"]:
            if w["time_valid"]:
                ax[2].broken_barh([(w["start"],w["end"]-w["start"])],(0,1),color="#217f85",alpha=.7)
            else: ax[2].plot(w["start"],.5,"rx",ms=3)
        ax[2].set_ylabel("Word support"); ax[2].set_yticks([]); ax[2].set_xlabel("Seconds from common clip origin")
        ax[2].set_xlim(0,s["media"]["duration"])
        fig.suptitle(s["id"]+" | automatic review; word-boundary truth not annotated")
        for a0 in ax: a0.grid(axis="x",alpha=.2)
        plot=s["key"]+".png"; fig.savefig(out/plot,dpi=130); plt.close(fig)
        with np.load(s["directory"]/"frames.npz") as z: pts=z["pts"]
        cap=cv2.VideoCapture(s["video"]); frames=[]
        try:
            for j,t in enumerate(np.linspace(s["media"]["video_start"],s["media"]["video_end"],5)[1:-1]):
                idx=int(np.argmin(abs(pts-t))); cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,im=cap.read()
                if not ok: raise ValueError(f"Cannot reread frame {idx}")
                image_name=s["key"]+f"_frame{j}.jpg"
                h,w=im.shape[:2]; im=cv2.resize(im,(round(w*min(1,480/w)),round(h*min(1,480/w))))
                cv2.imwrite(str(out/image_name),im)
                frames.append({"image":image_name,"frame_index":idx,"pts":float(pts[idx])})
                contacts.append((s["id"],float(pts[idx]),cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
        finally: cap.release()
        if s["media"]["silent_audio"]: category="digital_silence_confirmed_by_waveform"
        elif not speech.any(): category="no_speech_detected_by_vad_requires_listening"
        elif float(row.recall)<.5: category="low_speech_time_coverage_requires_word_boundary_review"
        else: category="structurally_invalid_words_require_review"
        note=notes.get(s["id"],{})
        if note and note["video_sha256"]!=s["video_sha256"]: raise ValueError("视觉复核笔记对应不同视频，需重新复核")
        excluded=np.isin(s["vision"]["frame_indices"],note.get("excluded_original_frames",[]))
        if note and not set(note["excluded_original_frames"]).issubset(set(s["vision"]["frame_indices"].tolist())):
            raise ValueError("视觉复核排除帧不在原生采样中")
        quality_dir=out/"quality"; quality_dir.mkdir(exist_ok=True)
        atomic_npz(quality_dir/(s["key"]+".npz"),original_frame_indices=s["vision"]["frame_indices"],
                   original_pts=s["vision"]["pts"],original_quality=s["vision"]["quality"],
                   review_excluded_mask=excluded,usable_after_review_mask=(s["vision"]["quality"]>0)&~excluded)
        cases.append({"sample_id":s["id"],"key":s["key"],"transcript":s["raw_text"],
                      "audio":quote(os.path.relpath(s["directory"]/"audio.wav",out)),
                      "video":quote(os.path.relpath(s["video"],out)),
                      "audio_offset":s["media"]["audio_offset"],"video_start":s["media"]["video_start"],
                      "duration":s["media"]["duration"],"audio_peak":s["media"]["audio_peak"],
                      "review_reasons":row.review_reasons,"automatic_category":category,
                      "vad_precision":row.precision,"vad_recall":row.recall,
                      "plot":plot,"frames":frames,"words":per_word,"human_status":"unannotated",
                      "visual_note":note.get("visual_note","未进行图像复核"),
                      "review_excluded_frames":int(excluded.sum()),"quality_sidecar":"quality/"+s["key"]+".npz"})
    pd.DataFrame(word_rows).to_csv(out/"word_annotations_template.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{k:v for k,v in c.items() if k not in ("words","frames")} for c in cases]).to_csv(out/"cases.csv",index=False,encoding="utf-8-sig")
    for page in range((len(cases)+5)//6):
        group=contacts[page*18:(page+1)*18]
        fig,axes=plt.subplots(6,3,figsize=(12,14),layout="constrained")
        for ax,entry in zip(axes.ravel(),group):
            sid,t,im=entry; ax.imshow(im); ax.set_title(sid+f" | {t:.3f}s",fontsize=8); ax.axis("off")
        for ax in axes.ravel()[len(group):]: ax.axis("off")
        fig.savefig(out/f"contact_{page+1}.jpg",dpi=120); plt.close(fig)
    payload=json.dumps(clean_json(cases),ensure_ascii=False).replace("</","<\\/")
    template=Path(__file__).with_name("review_template.html").read_text()
    (out/"index.html").write_text(template.replace("__CASES_JSON__",payload),encoding="utf-8")
    report={"samples":len(cases),"words_for_annotation":len(word_rows),
            "automatic_categories":pd.Series([c["automatic_category"] for c in cases]).value_counts().to_dict(),
            "human_annotated_words":0,"human_boundary_mae_seconds":None,
            "visual_reviewed_cases":len(notes),"review_excluded_native_frames":sum(c["review_excluded_frames"] for c in cases),
            "visual_review_scope":"AI查看每段3帧，另查看鸟类片段全部8个检测框；其余帧未逐帧确认。",
            "quality_sidecars":"只屏蔽有图像证据的8个误报。原主流程特征与本轮实验输入未应用该事后复核掩码。",
            "note":"自动检查与素材整理已完成；未把自动词边界冒充人工真值。人工填写模板后运行score_annotations。",
            "provenance":provenance(cfg,[__file__,data.__file__,Path(__file__).with_name("review_template.html")]+([notes_path] if notes_path.exists() else []))}
    atomic_json(out/"report.json",report); atomic_json(out/"cases.json",clean_json(cases))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def score_annotations(path,cfg=None):
    cfg=cfg or default_config(); out=output_dir(cfg,"review")
    table=pd.read_csv(path)
    reference=pd.read_csv(out/"word_annotations_template.csv")
    keys=["sample_id","word_index"]
    if table.duplicated(keys).any(): raise ValueError("人工标注出现重复词")
    known=pd.MultiIndex.from_frame(reference[keys])
    if not pd.MultiIndex.from_frame(table[keys]).isin(known).all(): raise ValueError("人工标注含未知样本或词索引")
    for column in ("manual_start","manual_end"):
        table[column]=pd.to_numeric(table[column],errors="raise")
    table=reference[keys+["auto_start","auto_end","auto_valid"]].merge(
        table[keys+["manual_start","manual_end","reviewer_code"]],on=keys,how="left",validate="one_to_one")
    if (table.manual_start.notna()^table.manual_end.notna()).any(): raise ValueError("人工词边界必须同时填写起点与终点")
    valid=table.manual_start.notna()&table.manual_end.notna()
    if np.any(table.loc[valid,"manual_end"]<=table.loc[valid,"manual_start"]): raise ValueError("人工词边界时长必须为正")
    if table.loc[valid,"reviewer_code"].fillna("").astype(str).str.strip().eq("").any(): raise ValueError("需填写匿名复核者代号")
    duration={c["sample_id"]:c["duration"] for c in json.loads((out/"cases.json").read_text())}
    for r in table[valid].itertuples():
        if not 0<=r.manual_start<r.manual_end<=duration[r.sample_id]: raise ValueError("人工边界超出片段")
    comparable=valid&table.auto_valid
    if comparable.any():
        e=np.abs(table.loc[comparable,["auto_start","auto_end"]].to_numpy()-table.loc[comparable,["manual_start","manual_end"]].to_numpy())
        scores={"boundary_mae_seconds":float(e.mean()),"boundary_within_100ms":float((e<=.1).mean()),"boundary_p90_seconds":float(np.percentile(e,90))}
    else: scores={"boundary_mae_seconds":None,"boundary_within_100ms":None,"boundary_p90_seconds":None}
    report={"annotated_words":int(valid.sum()),"comparable_auto_valid_words":int(comparable.sum()),
            "annotated_auto_invalid_words":int((valid&~table.auto_valid).sum()),**scores}
    atomic_json(out/"human_scores.json",report); return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--annotations")
    a=p.parse_args()
    print(json.dumps(score_annotations(a.annotations),ensure_ascii=False)) if a.annotations else review()
