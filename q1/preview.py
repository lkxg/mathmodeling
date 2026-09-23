from pathlib import Path

import numpy as np
import soundfile as sf

from .common import read_json
from .pipeline import Cache


def render(cfg, sample, start=0, end=None):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Sample IDs contain the literal separator "$_$"; they are not math text.
    plt.rcParams["text.parse_math"] = False

    cache = Cache(cfg)
    if not cache.valid(sample, "aggregate"):
        raise ValueError("此样本尚无通过当前配置/代码校验的完整特征")
    directory = cache.directory(sample)
    media = read_json(directory / "media.json")
    words = read_json(directory / "align.json")["words"]
    end = min(media["duration"], start + 12) if end is None else min(end, media["duration"])
    if not 0 <= start < end:
        raise ValueError("预览区间无效")
    fig = plt.figure(figsize=(17, 10), layout="constrained")
    grid = fig.add_gridspec(5, 6, height_ratios=[1.5, 1, 1, 1, 2])
    axes = [fig.add_subplot(grid[i, :]) for i in range(4)]
    for ax in axes:
        ax.set_xlim(start, end)
        ax.grid(axis="x", alpha=.2)
    for i, word in enumerate(words):
        if word["start"] < end and word["end"] > start:
            color = "#2a9d8f" if word["time_valid"] else "#e76f51"
            axes[0].broken_barh([(word["start"], word["end"] - word["start"])], (0, 1), facecolors=color, alpha=.35)
            axes[0].text((word["start"] + word["end"]) / 2, .08 + .45 * (i % 2), word["text"],
                         rotation=35, ha="center", va="bottom", fontsize=8, clip_on=True)
    axes[0].set_ylim(0, 1.8)
    axes[0].set_yticks([])
    axes[0].set_ylabel("Words")
    wave, sr = sf.read(directory / "audio.wav")
    t = np.arange(len(wave)) / sr + media["audio_offset"]
    keep = (t >= start) & (t <= end)
    stride = max(1, int(keep.sum()) // 20000)
    axes[1].plot(t[keep][::stride], wave[keep][::stride], linewidth=.4)
    axes[1].set_ylabel("Waveform")
    am = read_json(directory / "audio_features.json")
    with np.load(directory / "audio_features.npz", allow_pickle=False) as a:
        idx = next((i for i, name in enumerate(am["acoustic_names"]) if "F0" in name), 0)
        axes[2].plot(a["acoustic_intervals"].mean(1), a["acoustic"][:, idx], linewidth=.8)
        axes[2].set_ylabel(am["acoustic_names"][idx], fontsize=8)
    with np.load(directory / "vision.npz", allow_pickle=False) as v:
        ts = v["pts"]
        au = v["features"][:, 10].astype(float)
        au[v["quality"] <= 0] = np.nan
        axes[3].plot(ts, au, ".-", markersize=2, linewidth=.8)
        axes[3].set_ylabel("OpenFace AU output 0")
        axes[3].set_xlabel("Seconds relative to the clip origin")
        frame_indices = v["frame_indices"]
    cap = cv2.VideoCapture(sample["video"])
    try:
        for column, when in enumerate(np.linspace(start, end, 8)[1:-1]):
            j = int(np.argmin(abs(ts - when)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[j]))
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"无法回读原视频帧 {frame_indices[j]}")
            ax = fig.add_subplot(grid[4, column])
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title(f"frame {frame_indices[j]}\n{ts[j]:.3f} s", fontsize=9)
            ax.axis("off")
    finally:
        cap.release()
    fig.suptitle(sample["id"] + " | alignment review (not sentiment predictions)")
    out = cache.root / "previews" / (sample["key"] + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
