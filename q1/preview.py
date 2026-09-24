
import numpy as np
import soundfile as sf

from .common import read_json
from .pipeline import Cache, reviewed_quality


def render(cfg, sample, start=0, end=None):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cache = Cache(cfg)
    if not cache.valid(sample, "aggregate"):
        raise ValueError("此样本尚无通过当前配置/代码校验的完整特征")
    directory = cache.directory(sample)
    media = read_json(directory / "media.json")
    words = read_json(directory / "align.json")["words"]
    feature_path = cache.files(sample, "aggregate")[0]
    with np.load(feature_path, allow_pickle=False) as data:
        intervals, time_valid = data["timestamps"], data["time_valid_mask"]
        word_ids, coverage = data["word_indices"], data["coverage"]
    end = min(media["duration"], start + 12) if end is None else min(end, media["duration"])
    if not 0 <= start < end:
        raise ValueError("预览区间无效")
    fig = plt.figure(figsize=(17, 12), layout="constrained")
    grid = fig.add_gridspec(6, 6, height_ratios=[1.6, 1, 1, 1, .8, 2])
    axes = [fig.add_subplot(grid[i, :]) for i in range(5)]
    for ax in axes:
        ax.set_xlim(start, end)
        ax.grid(axis="x", alpha=.2)
    for i, word in enumerate(words):
        if word["start"] < end and word["end"] > start:
            color = "#2a9d8f" if word["time_valid"] else "#e76f51"
            axes[0].broken_barh([(word["start"], word["end"] - word["start"])], (0, 1), facecolors=color, alpha=.35)
            axes[0].text((word["start"] + word["end"]) / 2, .08 + .45 * (i % 2), word["text"],
                         rotation=35, ha="center", va="bottom", fontsize=8, clip_on=True, parse_math=False)
            if not word["time_valid"] and word["end"] <= word["start"]:
                axes[0].vlines((word["start"] + word["end"]) / 2, 0, 1, color="#e76f51", linewidth=1)
    for a, b in intervals[(word_ids < 0) & time_valid]:
        if a < end and b > start:
            axes[0].broken_barh([(a, b - a)], (-.35, .22), facecolors="#8b99a2", alpha=.65)
    axes[0].set_ylim(-.45, 2.0)
    axes[0].set_yticks([])
    axes[0].set_ylabel("Words / gaps")
    axes[0].legend(handles=[Patch(color="#2a9d8f", alpha=.35, label="Time-valid word"),
                           Patch(color="#e76f51", alpha=.35, label="Invalid word time"),
                           Patch(color="#8b99a2", alpha=.65, label="Unassigned gap (not a silence label)")],
                   loc="upper right", ncol=3, fontsize=8, framealpha=.9)
    wave, sr = sf.read(directory / "audio.wav")
    t = np.arange(len(wave)) / sr + media["audio_offset"]
    keep = (t >= start) & (t <= end)
    stride = max(1, int(keep.sum()) // 20000)
    axes[1].plot(t[keep][::stride], wave[keep][::stride], linewidth=.4)
    axes[1].set_ylabel("Waveform")
    am = read_json(directory / "audio.json")
    with np.load(directory / "audio.npz", allow_pickle=False) as a:
        idx = next((i for i, name in enumerate(am["acoustic_names"]) if "F0" in name), 0)
        values = a["acoustic"][:, idx].astype(float)
        values[~a["acoustic_valid"]] = np.nan
        axes[2].plot(a["acoustic_intervals"].mean(1), values, linewidth=.8)
        axes[2].set_ylabel(am["acoustic_names"][idx], fontsize=8)
    with np.load(directory / "vision.npz", allow_pickle=False) as v:
        ts = v["pts"]
        au = v["features"][:, 10].astype(float)
        quality, _, _ = reviewed_quality(sample, v, cfg)
        au[quality <= 0] = np.nan
        axes[3].plot(ts, au, ".-", markersize=2, linewidth=.8)
        axes[3].set_ylabel("OpenFace AU output 0")
        frame_indices = v["frame_indices"]
    for column, (color, name, style) in enumerate([("#d67b30", "Audio support", "--"), ("#147d86", "Visual support", "-")]):
        xs, ys = [], []
        for k in np.flatnonzero(time_valid):
            a, b = intervals[k]
            if a < end and b > start:
                xs.extend([a, b, np.nan]); ys.extend([coverage[k, column], coverage[k, column], np.nan])
        axes[4].plot(xs, ys, color=color, linestyle=style, linewidth=1.5, label=name)
    axes[4].set(ylim=(-.05, 1.08), yticks=[0, .5, 1], ylabel="Coverage", xlabel="Seconds relative to the clip origin")
    axes[4].legend(loc="lower right", ncol=2, fontsize=8)
    cap = cv2.VideoCapture(sample["video"])
    try:
        for column, when in enumerate(np.linspace(start, end, 8)[1:-1]):
            j = int(np.argmin(abs(ts - when)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[j]))
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"无法回读原视频帧 {frame_indices[j]}")
            ax = fig.add_subplot(grid[5, column])
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(f"frame {frame_indices[j]} | q={quality[j]:.3f}\n{ts[j]:.3f} s", fontsize=9)
            ax.axis("off")
    finally:
        cap.release()
    # The literal ID separator "$_$" must not be parsed as a math expression.
    fig.suptitle(sample["id"] + " | SAQW-TA: word/gap timeline and native observations", parse_math=False)
    out = cache.root / "report" / "figures" / (sample["key"] + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
