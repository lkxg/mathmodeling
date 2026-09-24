from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager

FONT="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams.update({"font.family":font_manager.FontProperties(fname=FONT).get_name(),
    "font.size":12,"axes.unicode_minus":False,"text.parse_math":False,
    "axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})
TEAL="#147D86"; GOLD="#D68A3A"; BLUE="#416B9A"; INK="#173644"; GRAY="#788B94"


def save(fig,path):
    fig.savefig(path,dpi=220,bbox_inches="tight",pad_inches=.12)
    plt.close(fig);return path


def box(ax,x,y,w,h,title,body,color=TEAL):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.015,rounding_size=.05",fc="#F0F7F8",ec=color,lw=1.5))
    ax.text(x+w/2,y+h*.73,title,ha="center",va="center",fontsize=14,color=color,weight="bold")
    ax.text(x+w/2,y+h*.34,body,ha="center",va="center",fontsize=11.3,color=INK,linespacing=1.5)


def arrow(ax,a,b,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=15,lw=1.6,color=color))


def architecture(out):
    fig,ax=plt.subplots(figsize=(10.5,9)); ax.set(xlim=(0,10),ylim=(0,9));ax.axis("off")
    box(ax,1,7.85,8,1,"附件1：100条原视频＋已有英文转写","样本清单 / 标签关联 / 文件SHA256 / 原始素材完整保留")
    box(ax,1,6.45,8,1,"共同时间坐标与源身份","原帧PTS · 音频采样偏移 · [0, D]秒时间轴 · 原始索引 / 有效性")
    arrow(ax,(5,7.85),(5,7.47))
    xs=[.1,3.45,6.8]
    labels=[("文本","原文＋音频强制对齐\nModernBERT词表示：768维"),
            ("语音","16 kHz单声道\nemotion2vec：768；LLD：25"),
            ("视觉","真实PTS采样 / 检测与跟踪\n单帧28维＋检测质量")]
    for x,(title,body) in zip(xs,labels):
        box(ax,x,4.55,3.05,1.25,title,body)
        arrow(ax,(5,6.45),(x+1.525,5.85));arrow(ax,(x+1.525,4.55),(5,4.0))
    box(ax,1,2.9,8,1.1,"主方案：词区间＋未分配间隙的可解释聚合","时间重叠长度 × 有效质量 → 归一化权重 → 加权均值 / 标准差")
    box(ax,.2,1.05,4.65,1.3,"正式输出与独立核验","768 / 818 / 56；掩码；CSR映射\n全量覆盖 / 源帧回溯 / 数值重建")
    box(ax,5.15,1.05,4.65,1.3,"补充实验（独立保存）","维度消融 / 固定锚点 / 采样扰动\n原生编码＋局部软对齐对照",GOLD)
    arrow(ax,(5,2.9),(2.5,2.4));arrow(ax,(5,2.9),(7.5,2.4),GOLD)
    ax.text(5,.45,"原生源序列始终保留；学习到的注意力关联不改写物理时间身份",ha="center",color=INK,fontsize=12)
    return save(fig,out/"01_architecture.png")


def timeline(out):
    fig,ax=plt.subplots(figsize=(10.5,4.6))
    ax.set(xlim=(0,6),ylim=(-.7,4.9),yticks=[0,1.3,2.6,3.9],yticklabels=["输出锚点","原生视觉","原生语音","原生词序列"],xlabel="共同时间轴 / 秒")
    intervals=[(.1,.75),(1.05,1.72),(2.1,3.2),(3.7,4.4),(4.8,5.6)]
    for i,(a,b) in enumerate(intervals):
        ax.broken_barh([(a,b-a)],(3.65,.48),facecolors=TEAL,alpha=.75);ax.text((a+b)/2,3.9,f"词{i+1}",ha="center",va="center",color="white")
    for x in np.arange(.03,5.96,.12):ax.plot([x,x],[2.37,2.83],color=BLUE,lw=2)
    for x in np.arange(.08,5.96,.43):ax.plot([x,x],[1.07,1.53],color=GOLD,lw=3)
    edges=sorted(set([0,6]+[v for ab in intervals for v in ab]))
    for a,b in zip(edges[:-1],edges[1:]):
        word=any(abs(a-u)<1e-8 and abs(b-v)<1e-8 for u,v in intervals)
        ax.add_patch(Rectangle((a,-.23),b-a,.46,fc=TEAL if word else "#DCE5E8",ec="white"))
    ax.axvspan(2.1,3.2,color=TEAL,alpha=.09)
    ax.text(2.65,4.5,"同一目标区间按重叠取源观测",ha="center",color=TEAL)
    ax.text(4.7,.48,"浅灰：未分配间隙",ha="center",fontsize=11,color=GRAY)
    ax.grid(axis="x",alpha=.2);fig.tight_layout()
    return save(fig,out/"02_timeline.png")


def data_overview(out,summary,probe):
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout="constrained")
    axes[0].hist(summary.duration,bins=[0,3,6,9,12,15,20,25,30],color=TEAL,edgecolor="white")
    axes[0].set(xlabel="原视频片段时长 / 秒",ylabel="样本数",title="片段长度分布")
    values=[probe["class_counts"][str(i)] for i in (-1,0,1)]
    axes[1].bar(["负向","中性","正向"],values,color=[BLUE,GRAY,GOLD])
    for i,y in enumerate(values):axes[1].text(i,y+.7,str(y),ha="center")
    axes[1].set(ylabel="样本数",title="附件1标签分布（仅用于辅助实验）",ylim=(0,65))
    return save(fig,out/"03_data.png")


def typical(out,s):
    import cv2,soundfile as sf
    fig=plt.figure(figsize=(11,8.4),layout="constrained")
    grid=fig.add_gridspec(5,3,height_ratios=[1.05,1,1,1,1.7])
    axes=[fig.add_subplot(grid[i,:]) for i in range(4)]
    for i,w in enumerate(s["words"]):
        if w["time_valid"]:axes[0].broken_barh([(w["start"],w["end"]-w["start"])],(0,.65),color=TEAL,alpha=.3)
        else:axes[0].plot(w["start"],.2,"rx",ms=8)
        axes[0].text((w["start"]+w["end"])/2,.75 if i%2==0 else 1.18,str(i+1),ha="center",fontsize=11,color=INK)
    axes[0].set(ylim=(-.1,1.6),yticks=[],ylabel="词序号")
    wave,sr=sf.read(s["directory"]/"audio.wav"); stride=max(1,len(wave)//15000)
    axes[1].plot((np.arange(len(wave))/sr+s["media"]["audio_offset"])[::stride],wave[::stride],color=BLUE,lw=.5)
    axes[1].set_ylabel("音频波形")
    names=json.loads((s["directory"]/"audio_features.json").read_text())["acoustic_names"]
    idx=next(i for i,n in enumerate(names) if n.startswith("F0"))
    axes[2].plot(s["acoustic"]["intervals"].mean(1),s["acoustic"]["x"][:,idx],color=BLUE,lw=1)
    axes[2].set_ylabel("F0半音值")
    av=s["vision"]["x"][:,10].astype(float);av[s["vision"]["quality"]<=0]=np.nan
    axes[3].plot(s["vision"]["pts"],av,".-",color=GOLD,lw=1,ms=3)
    axes[3].set(ylabel="AU原始输出0",xlabel="共同时间轴 / 秒")
    for ax in axes:ax.set_xlim(0,s["media"]["duration"]);ax.grid(axis="x",alpha=.2)
    cap=cv2.VideoCapture(s["video"])
    for j,t in enumerate([.8,2.4,4.7]):
        vi=int(np.argmin(abs(s["vision"]["pts"]-t))); fi=int(s["vision"]["frame_indices"][vi])
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,im=cap.read();assert ok
        ax=fig.add_subplot(grid[4,j]);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.axis("off")
        ax.set_title(f"原帧 {fi}｜{s['vision']['pts'][vi]:.3f} 秒",fontsize=12)
    cap.release();return save(fig,out/"04_typical.png")


def probe_figure(out,probe):
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.6),layout="constrained")
    sets=[(["audio_768_aligned","audio_793_aligned","audio_818_aligned"],["768","793","818"],"音频维度"),
          (["vision_28_aligned","vision_56_aligned"],["28","56"],"视觉维度"),
          (["text_modernbert","fusion_aligned","fusion_clip"],["文本","词位置融合","整段融合"],"文本与融合")]
    for ax,(names,labels,title) in zip(axes,sets):
        values=[probe["results"][n]["macro_f1_3"] for n in names]
        ax.bar(labels,values,color=[TEAL,BLUE,GOLD][:len(values)]);ax.set(ylim=(0,.6),title=title,ylabel="三分类 Macro-F1")
        for j,v in enumerate(values):ax.text(j,v+.01,f"{v:.3f}",ha="center",fontsize=10)
        ax.tick_params(axis="x",labelsize=10);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    return save(fig,out/"05_probe.png")


def anchors_figure(out,r):
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout="constrained")
    modes=r["protocol"]["modes"];labels=["词＋间隙","0.25秒","0.5秒","1秒","50段"]
    for ax,metric,title in zip(axes,["mae","macro_f1_3"],["回归 MAE（越低越好）","三分类 Macro-F1"]):
        for pool,name,color in [("uniform","锚点等权",TEAL),("duration","时长加权",GOLD)]:
            ax.plot(labels,[r["results"][m+"__"+pool][metric] for m in modes],"o-",label=name,color=color)
        ax.set_title(title);ax.grid(alpha=.2);ax.legend(fontsize=10)
    return save(fig,out/"06_anchors.png")


def sampling_figure(out,r):
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),layout="constrained");rates=[5,8,10,12,25]
    for ax,field,scale,title in [(axes[0],"mean_abs_centroid_shift_seconds",1000,"代表帧时间偏移中位数 / ms"),(axes[1],"relative_feature_change",1,"视觉特征相对变化中位数")]:
        ax.plot(rates,[scale*r["summary"][str(n)][field] for n in rates],"o-",color=TEAL)
        ax.set(xlabel="目标采样率 / Hz",title=title,xticks=rates);ax.grid(alpha=.2)
    return save(fig,out/"07_sampling.png")


def soft_figures(out,r,root):
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0),layout="constrained")
    names=list(r["results"]);labels=["硬重叠","时间核","局部注意力","局部无惩罚","全局注意力"]
    for ax,metric,title in zip(axes,["mae","macro_f1_3"],["回归 MAE（越低越好）","三分类 Macro-F1"]):
        ax.bar(np.arange(5),[r["results"][n][metric]["mean"] for n in names],yerr=[r["results"][n][metric]["seed_std"] for n in names],color=[TEAL,BLUE,GOLD,"#B29670",GRAY],capsize=3)
        ax.set(xticks=np.arange(5),xticklabels=labels,title=title);ax.tick_params(axis="x",labelrotation=20,labelsize=10)
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    p1=save(fig,out/"08_soft_results.png")
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),layout="constrained")
    for ax,mode,title in zip(axes,["hard_overlap","local_attention"],["硬重叠：固定物理权重","局部注意力：受限的学习关联"]):
        with np.load(root/"soft/attention"/mode/"-3g5yACwYnA__13.npz") as z:
            mesh=ax.pcolormesh(z["vision_source_intervals"].mean(1),z["anchor_times"].mean(1),z["vision_weights"],shading="nearest",cmap="viridis")
            ax.set(xlabel="视觉源时间 / 秒",ylabel="查询锚点时间 / 秒",title=title);fig.colorbar(mesh,ax=ax,shrink=.8)
    p2=save(fig,out/"09_attention.png");return p1,p2


def quality_figure(out,s):
    import cv2
    frames=json.loads((s["directory"]/"vision.json").read_text())["frames"]
    good=np.flatnonzero(s["vision"]["quality"]>0)
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.2),layout="constrained")
    cap=cv2.VideoCapture(s["video"])
    for ax,i in zip(axes.ravel(),good[[0,2,5,7]]):
        f=frames[i];cap.set(cv2.CAP_PROP_POS_FRAMES,f["frame_index"]);ok,im=cap.read();assert ok
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));l,t,r,b=f["bbox"]
        ax.add_patch(Rectangle((l,t),r-l,b-t,fill=False,ec="#E23B33",lw=2.3));ax.axis("off")
        ax.set_title(f"原帧{f['frame_index']}｜PTS {f['pts']:.3f}秒",fontsize=12)
    cap.release();return save(fig,out/"10_false_face.png")


def vad_figure(out,root):
    table=pd.read_csv(root.parent/"alignment_vad.csv")
    fig,ax=plt.subplots(figsize=(10.5,3.5),layout="constrained")
    for flag,label,color in [(False,"未列入复核",TEAL),(True,"列入18条复核集合",GOLD)]:
        values=table.loc[table.needs_review==flag,"recall"].dropna()
        ax.hist(values,bins=np.linspace(0,1,11),alpha=.72,label=label,color=color)
    ax.set(xlabel="VAD语音被有效词区间覆盖的比例",ylabel="样本数",title="VAD一致性筛查：96条可评分样本")
    ax.legend(fontsize=11);return save(fig,out/"11_vad.png")


def build(out,samples,summary,probe,reports,root):
    out.mkdir(parents=True,exist_ok=True)
    files=[architecture(out),timeline(out),data_overview(out,summary,probe),typical(out,samples[0]),
           probe_figure(out,probe),anchors_figure(out,reports["anchors"]),sampling_figure(out,reports["sampling"])]
    files+=list(soft_figures(out,reports["soft"],root))
    files+=[quality_figure(out,next(s for s in samples if s["key"]=="-NFrJFQijFE__2")),vad_figure(out,root)]
    return {p.stem.split('_',1)[1]:p for p in files}
