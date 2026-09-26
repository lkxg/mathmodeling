"""Small grouped feature probe and descriptive figures; leaves Q1 features intact."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .common import atomic_json, file_hash, load_config, read_json

SEED = 2026
LABELS = [-1, 0, 1]
NAMES = {-1: "负向", 0: "中性", 1: "正向"}
MODALITIES = ("text", "audio", "vision")
TITLES = {"majority": "多数类基线", "text": "文本 T", "audio": "语音 A", "vision": "视觉 V", "fusion": "拼接 T+A+V"}
COLORS = {-1: "#B54D37", 0: "#79838B", 1: "#147D86"}


def pooled_features(root, samples):
    blocks = {m: [] for m in MODALITIES}
    available, hashes = [], {}
    for sample in samples:
        path = root / "features" / (sample["key"] + ".npz")
        hashes[str(path.relative_to(root))] = file_hash(path)
        with np.load(path, allow_pickle=False) as z:
            assert [z[m].shape[1] for m in MODALITIES] == [768, 50, 56]
            durations = z["timestamps"][:, 1] - z["timestamps"][:, 0]
            row_available = []
            for j, modality in enumerate(MODALITIES):
                mask = z["valid_mask"] & z["modality_mask"][:, j]
                if modality == "text":
                    mask &= z["word_indices"] >= 0
                    weights = np.ones(int(mask.sum()), dtype=np.float64)
                else:
                    mask &= z["time_valid_mask"] & np.isfinite(durations) & (durations > 0)
                    weights = durations[mask]
                row_available.append(bool(mask.any()))
                values = z[modality][mask].astype(np.float64)
                assert np.isfinite(values).all()
                pooled = np.average(values, axis=0, weights=weights) if mask.any() else np.full(z[modality].shape[1], np.nan)
                blocks[modality].append(pooled)
            available.append(row_available)
    blocks = {m: np.stack(rows) for m, rows in blocks.items()}
    blocks["fusion"] = np.concatenate([blocks[m] for m in MODALITIES], axis=1)
    return blocks, np.asarray(available, bool), hashes


def score(y, prediction):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    return {"accuracy": float(accuracy_score(y, prediction)),
            "macro_f1": float(f1_score(y, prediction, labels=LABELS, average="macro", zero_division=0)),
            "per_class_f1": {str(k): float(v) for k, v in zip(LABELS, f1_score(y, prediction, labels=LABELS, average=None, zero_division=0))},
            "confusion_matrix": confusion_matrix(y, prediction, labels=LABELS).tolist()}


def grouped_probe(blocks, y, groups):
    from sklearn.dummy import DummyClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(blocks["text"], y, groups))
    assignment = np.full(len(y), -1, dtype=int)
    fold_info = []
    for k, (train, test) in enumerate(folds):
        assert not (set(groups[train]) & set(groups[test]))
        assert np.all(assignment[test] == -1)
        assert set(y[train]) == set(LABELS)
        assignment[test] = k + 1
        fold_info.append({"fold": k + 1, "train_samples": len(train), "test_samples": len(test),
                          "train_groups": len(set(groups[train])), "test_groups": len(set(groups[test])),
                          "train_class_counts": {str(c): int((y[train] == c).sum()) for c in LABELS},
                          "test_class_counts": {str(c): int((y[test] == c).sum()) for c in LABELS},
                          "test_video_ids": sorted(set(groups[test]))})
    assert np.all(assignment > 0)
    metrics, predictions = {}, {}
    for modality in ["majority", *blocks]:
        X = np.zeros((len(y), 1)) if modality == "majority" else blocks[modality]
        pred = np.full(len(y), 99, dtype=int)
        fold_scores = []
        for k, (train, test) in enumerate(folds):
            model = DummyClassifier(strategy="most_frequent") if modality == "majority" else make_pipeline(
                SimpleImputer(strategy="mean", keep_empty_features=True), StandardScaler(),
                LogisticRegression(C=1., solver="lbfgs", class_weight="balanced", max_iter=5000, random_state=SEED))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(X[train], y[train])
            pred[test] = model.predict(X[test])
            fold_scores.append({"fold": k + 1, **score(y[test], pred[test])})
        assert set(pred) <= set(LABELS)
        predictions[modality] = pred
        metrics[modality] = {**score(y, pred), "fold_scores": fold_scores,
            "fold_accuracy_mean": float(np.mean([v["accuracy"] for v in fold_scores])),
            "fold_accuracy_sd": float(np.std([v["accuracy"] for v in fold_scores], ddof=1)),
            "fold_macro_f1_mean": float(np.mean([v["macro_f1"] for v in fold_scores])),
            "fold_macro_f1_sd": float(np.std([v["macro_f1"] for v in fold_scores], ddof=1))}
    return metrics, predictions, assignment, fold_info


def plotting_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def probe_figures(out, blocks, available, y, metrics):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    plt = plotting_setup()
    fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
    keys = list(metrics)
    x = np.arange(len(keys))
    for offset, metric, name, color in [(-.19, "accuracy", "Accuracy", "#147D86"), (.19, "macro_f1", "Macro-F1", "#D58C45")]:
        bars = ax.bar(x + offset, [metrics[k][metric] for k in keys], width=.36, color=color, label=name)
        ax.bar_label(bars, labels=[f"{metrics[k][metric]:.1%}" for k in keys], padding=3, fontsize=10)
    ax.set(xticks=x, xticklabels=[TITLES[k] for k in keys], ylim=(0, 1), ylabel="100条折外预测汇总")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.15)
    fig.savefig(out / "probe_metrics.png", dpi=210, facecolor="white"); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    coords, details = {}, {}
    for ax, (modality, X) in zip(axes.flat, blocks.items()):
        # This descriptive visualization uses all features, but never labels during fitting.
        X = SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(X)
        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(50, X.shape[1], len(X)-1), svd_solver="full", random_state=SEED)
        reduced = pca.fit_transform(X)
        tsne = TSNE(n_components=2, perplexity=15., max_iter=1000, init="pca", learning_rate="auto", random_state=SEED, n_jobs=1)
        xy = tsne.fit_transform(reduced)
        coords[modality] = xy
        missing = ~available.all(1) if modality == "fusion" else ~available[:, MODALITIES.index(modality)]
        for label in LABELS:
            idx = y == label
            ax.scatter(xy[idx, 0], xy[idx, 1], s=30, color=COLORS[label], alpha=.8,
                       edgecolors="white", linewidths=.35, label=f"{NAMES[label]} ({idx.sum()})")
        ax.scatter(xy[missing, 0], xy[missing, 1], s=56, c="black", marker="x", linewidths=1.2)
        ax.set(title=f"{TITLES[modality]} | n=100，整段缺失填补 {int(missing.sum())}条", xticks=[], yticks=[])
        ax.spines[["top", "right"]].set_visible(False)
        details[modality] = {"pca_components": pca.n_components_,
                            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
                            "tsne_kl_divergence": float(tsne.kl_divergence_), "imputed_samples": int(missing.sum())}
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[k], label=f"{NAMES[k]} ({int((y==k).sum())})") for k in LABELS]
    handles.append(Line2D([], [], marker="x", linestyle="", color="black", label="含整段缺失模态的填补样本"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=10)
    fig.savefig(out / "tsne.png", dpi=210, facecolor="white"); plt.close(fig)
    return coords, details


def diagnostic_figures(root, out, samples, cfg):
    import cv2
    import soundfile as sf
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle
    plt = plotting_setup()
    index = {r["key"]: r for r in samples}
    validation = read_json(root / "validation.json")["word_alignment"]
    values = [validation["exclusive_reasons"][k] for k in ("non_positive_duration", "silent_audio", "out_of_audio_bounds")]
    assert sum(values) == validation["invalid_words"] == 219
    fig, ax = plt.subplots(figsize=(10, 2.6), layout="constrained")
    bars = ax.barh(["非正时长", "数字静音", "越界"], values, color=["#147D86", "#D58C45", "#B54D37"], height=.55)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{n}词 / {n/219:.2%}" for n in values], padding=5)
    ax.set(xlim=(0, 240), xlabel="互斥分类词数（分母：219个结构无效词）")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "invalid_word_reasons.png", dpi=210, facecolor="white"); plt.close(fig)

    def source(key):
        sample = index[key]
        assert file_hash(sample["video"]) == sample["video_sha256"]
        directory = root / "cache" / key
        with np.load(directory / "vision.npz", allow_pickle=False) as z:
            vision = {k: z[k].copy() for k in z.files}
        with np.load(root / "features" / f"{key}.npz", allow_pickle=False) as z:
            exported = {k: z[k].copy() for k in z.files}
        return sample, directory, read_json(directory / "media.json"), read_json(directory / "vision.json"), vision, exported

    def draw_frame(ax, sample, metadata, row, title, box=False):
        frame = metadata["frames"][row]
        cap = cv2.VideoCapture(sample["video"])
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame["frame_index"])
            ok, image = cap.read()
            if not ok: raise ValueError(f"Cannot read source frame: {sample['key']}")
        finally:
            cap.release()
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if box and frame["bbox"] is not None:
            l,t,r,b = frame["bbox"]
            ax.add_patch(Rectangle((l,t),r-l,b-t,fill=False,color="#F04F43",linewidth=2.5))
        ax.set_title(f"{title}\nframe {frame['frame_index']} | PTS {frame['pts']:.3f}s", fontsize=11)
        ax.axis("off")
        return frame

    sample, directory, media, vmeta, vision, exported = source("-mJ2ud6oKI8__1")
    wave,sr = sf.read(directory / "audio.wav", dtype="float32")
    words = exported["word_indices"] >= 0
    assert media["audio_peak"] == float(np.abs(wave).max()) == 0
    assert not exported["modality_mask"][:,1].any()
    assert not exported["time_valid_mask"][words].any()
    fig = plt.figure(figsize=(11, 6.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])
    ax = fig.add_subplot(grid[0,0]); frame = draw_frame(ax,sample,vmeta,len(vmeta["frames"])//2,"数字静音样本的原视频帧")
    ax = fig.add_subplot(grid[0,1]); ax.axis("off")
    counts = exported["modality_mask"].sum(0).astype(int)
    details = (f"{sample['key']}\n\n音频峰值 = {media['audio_peak']:.1f}；RMS = {media['audio_rms']:.1f}\n"
        f"音频有效源窗：{int(exported['source_acoustic_valid'].sum())}/{len(exported['source_acoustic_valid'])}\n"
        f"结构有效词时间：{int(exported['time_valid_mask'][words].sum())}/{int(words.sum())}\n"
        f"文本有效位置：{counts[0]}/{len(words)}（全部{int(words.sum())}词保留）\n"
        f"音频 / 视觉有效位置：{counts[1]}/{len(words)}、{counts[2]}/{len(words)}\n\n"
        "该片段也没有有效人脸观测；\n视觉缺失由独立检测规则确定。")
    ax.text(0,.95,details,va="top",fontsize=11,linespacing=1.65)
    ax = fig.add_subplot(grid[1,0])
    step=max(1,len(wave)//5000)
    ax.plot(np.arange(len(wave))[::step]/sr+media["audio_offset"],wave[::step],color="#147D86",linewidth=1)
    for sign in [-1,1]: ax.axhline(sign*cfg['media']['silence_peak'],color="#D58C45",linestyle="--",linewidth=1)
    ax.set(xlabel="片段时间 / s",ylabel="波形幅值",ylim=(-1.3e-4,1.3e-4),title="原始波形与±1e−4数字静音阈值",xlim=(0,media['duration']))
    ax.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    ax = fig.add_subplot(grid[1,1])
    ax.imshow(exported['modality_mask'].T.astype(int),cmap=ListedColormap(['#EDF1F5','#147D86']),vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    ax.set(yticks=[0,1,2],yticklabels=['文本 T','音频 A','视觉 V'],xlabel='输出位置索引 i',title='正式modality_mask：绿色=可用，灰色=不可用')
    fig.savefig(out / 'silent_case.png',dpi=210,facecolor='white'); plt.close(fig)
    cases = {sample['key']:{'video_sha256':sample['video_sha256'],'duration':media['duration'],
        'audio_peak':media['audio_peak'],'audio_rms':media['audio_rms'],'positions':len(words),'words':int(words.sum()),
        'valid_word_times':int(exported['time_valid_mask'][words].sum()),'valid_modality_positions':counts.tolist(),
        'source_audio_windows':len(exported['source_acoustic_valid']),'valid_source_audio_windows':int(exported['source_acoustic_valid'].sum()),
        'source_frame':frame['frame_index'],'source_pts':frame['pts']}}

    fig, axes = plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    for row,key in enumerate(['-NFrJFQijFE__1','-NFrJFQijFE__2']):
        sample,directory,media,vmeta,vision,exported = source(key)
        positive = np.flatnonzero(vision['quality']>0)
        j=int(positive[0]) if len(positive) else len(vision['quality'])//2
        frame = draw_frame(axes[row,0],sample,vmeta,j,'未获得有效人脸观测' if row==0 else '红框：原检测器落在羽毛上的误检',box=row==1)
        q = exported['source_vision_quality']; excluded=exported['source_vision_excluded']
        assert len(q)==len(vision['quality'])
        mapped=set(exported['map_vision_indices'].tolist())
        assert not (mapped & set(np.flatnonzero(excluded).tolist()))
        ax=axes[row,1]
        ax.plot(vision['pts'],vision['quality'],'.--',color='#B54D37',label='原检测质量',markersize=4)
        ax.plot(vision['pts'],q,'-',color='#147D86',linewidth=1.8,label='实际聚合质量')
        ax.axhline(.8,color='#C3A282',linestyle=':',linewidth=1,label='检测阈值0.8')
        ax.set(xlabel='原帧PTS / s',ylabel='质量权重 q',ylim=(-.08,1.06),
               title=f"{key}\n有效帧 {int((vision['quality']>0).sum())}→{int((q>0).sum())} / {len(q)}；视觉位置 {int(exported['modality_mask'][:,2].sum())}/{len(exported['modality_mask'])}")
        ax.legend(frameon=False,fontsize=8,loc='upper right')
        ax.grid(alpha=.15)
        cases[key]={'video_sha256':sample['video_sha256'],'sampled_frames':len(q),
            'raw_valid_frames':int((vision['quality']>0).sum()),'final_valid_frames':int((q>0).sum()),
            'excluded_original_frames':vision['frame_indices'][excluded].tolist(),
            'positions':len(exported['modality_mask']),'valid_vision_positions':int(exported['modality_mask'][:,2].sum()),
            'vision_csr_entries':len(exported['map_vision_indices']),'excluded_frames_present_in_map':False,
            'source_frame':frame['frame_index'],'source_pts':frame['pts'],'source_bbox':frame['bbox']}
    assert len(cases['-NFrJFQijFE__2']['excluded_original_frames'])==8
    fig.savefig(out/'vision_cases.png',dpi=210,facecolor='white'); plt.close(fig)
    return cases


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')))
    args=parser.parse_args(argv)
    cfg=load_config(args.config); root=Path(cfg['output_dir']); out=root/'sanity_check'
    out.mkdir(exist_ok=True); figs=out/'figures'; figs.mkdir(exist_ok=True)
    manifest=read_json(root/'manifest.json'); samples=manifest['samples']
    assert file_hash(cfg['labels'])==manifest['label_sha256']
    assert len(samples)==100 and len({r['key'] for r in samples})==100
    y=np.sign([r['label'] for r in samples]).astype(int)
    groups=np.asarray([r['video_id'] for r in samples])
    blocks,available,hashes=pooled_features(root,samples)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        metrics,predictions,assignment,folds=grouped_probe(blocks,y,groups)
        coordinates,projection=probe_figures(figs,blocks,available,y,metrics)
    cases=diagnostic_figures(root,figs,samples,cfg)
    asr={r['key']:r for r in read_json(root/'asr_audit/results.json')['records']}
    with (out/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['key','video_id','fold','polarity','asr_group','text_available','audio_available','vision_available']+[f'pred_{k}' for k in predictions]
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for i,r in enumerate(samples):
            a=asr[r['key']]
            group='not_comparable' if not a['wer_comparable'] else ('high_suspect' if a['wer']>.9 else 'not_flagged')
            writer.writerow({'key':r['key'],'video_id':r['video_id'],'fold':int(assignment[i]),'polarity':int(y[i]),'asr_group':group,
                **{f'{m}_available':bool(available[i,j]) for j,m in enumerate(MODALITIES)},
                **{f'pred_{k}':int(pred[i]) for k,pred in predictions.items()}})
    with (out/'projection.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f); writer.writerow(['key','polarity']+[f'{m}_{axis}' for m in blocks for axis in ('x','y')])
        for i,r in enumerate(samples):writer.writerow([r['key'],int(y[i])]+[float(v) for m in blocks for v in coordinates[m][i]])
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'samples':len(samples),'video_groups':len(set(groups)),
        'class_counts':{str(k):int((y==k).sum()) for k in LABELS},'class_order':LABELS,
        'modality_missing_samples':{m:int((~available[:,j]).sum()) for j,m in enumerate(MODALITIES)},
        'settings':{'seed':SEED,'folds':5,'split':'StratifiedGroupKFold(video_id), shuffled',
            'classifier':'L2 LogisticRegression','C':1.,'solver':'lbfgs','max_iter':5000,'class_weight':'balanced',
            'hyperparameter_search':False,'pooling':{'text':'mean over valid original-word positions','audio_vision':'duration-weighted mean over available exported positions'},
            'missing_modalities':'NaN pooled vector; training-fold mean imputation, no added mask features',
            'standardization':'training fold only','dimensions':{m:blocks[m].shape[1] for m in blocks},
            'tsne':{'perplexity':15,'max_iter':1000,'seed':SEED,'initialization':'pca','pca_dimensions':50,
                    'labels_used_during_fit':False,'scope':'all 100 features; descriptive only; not used for classifier evaluation or selection'}},
        'folds':folds,'metrics':metrics,'projection':projection,'anomaly_cases':cases,
        'versions':{n:importlib.metadata.version(n) for n in ['scikit-learn','numpy','scipy','matplotlib']},
        'feature_sha256':hashes,'source_sha256':{n:file_hash(root/n) for n in ['manifest.json','validation.json','asr_audit/results.json']},
        'labels_sha256':file_hash(cfg['labels']),'quality_review_sha256':file_hash(cfg['quality_review']),
        'code_sha256':file_hash(__file__),'all_100_samples_retained':True,'asr_filter_applied':False,
        'formal_features_modified':False,'interpretation':'Exploratory small-data feature probe; no universal 70% pass criterion; t-SNE is not proof of class separability.'}
    atomic_json(out/'results.json',result)
    assert all(file_hash(root/name)==value for name,value in hashes.items())
    print(json.dumps({'class_counts':result['class_counts'],'missing':result['modality_missing_samples'],
        'metrics':{k:{m:v[m] for m in ['accuracy','macro_f1']} for k,v in metrics.items()},'output':str(out)},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
