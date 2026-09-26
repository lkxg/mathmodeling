"""Non-neural RBF kernel baseline on pooled multimodal statistics.

This is a structurally different complete-input comparison, not a missing-input
model. All preprocessing is fitted on the training split only.
"""
from __future__ import annotations

import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .core import DATA, OUT, metrics, pack_split, read_pkl, save_json


def moments(values, mask):
    values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    count = np.maximum(mask.sum(1, keepdims=True), 1)
    mean = (values * mask[..., None]).sum(1) / count
    var = (((values - mean[:, None]) * mask[..., None]) ** 2).sum(1) / count
    present = mask.mean(1, keepdims=True)
    return np.concatenate((mean, np.sqrt(var + 1e-8), present), axis=1)


def features(raw, packed):
    mask = packed["mask"].numpy()
    result = []
    for key, modality in (("text", 0), ("audio", 1), ("vision", 2)):
        values = np.asarray(raw[key], dtype=np.float32) if key == "text" else packed[key].numpy()
        result.append(moments(values, mask[..., modality]))
    return result


def main():
    started = time.time()
    raw = read_pkl(DATA / "附件2-数据集特征文件/aligned_50.pkl")
    train, stats = pack_split(raw["train"], "cpu")
    valid, _ = pack_split(raw["valid"], "cpu", stats)
    train_features = features(raw["train"], train)
    valid_features = features(raw["valid"], valid)
    processors = [make_pipeline(StandardScaler(), PCA(n_components=dim,
                   svd_solver="randomized", random_state=2026), StandardScaler())
                  for dim in (96, 48, 24)]
    x_train = np.concatenate([p.fit_transform(x) / np.sqrt(dim)
                              for p, x, dim in zip(processors, train_features, (96, 48, 24))], axis=1)
    x_valid = np.concatenate([p.transform(x) / np.sqrt(dim)
                              for p, x, dim in zip(processors, valid_features, (96, 48, 24))], axis=1)
    classifier = SVC(C=3, kernel="rbf", gamma="scale", break_ties=True,
                     random_state=2026)
    regressor = SVR(C=1, epsilon=.1, kernel="rbf", gamma="scale")
    classifier.fit(x_train, train["class"].numpy())
    regressor.fit(x_train, train["reg"].numpy())
    logits = classifier.decision_function(x_valid)
    reg = regressor.predict(x_valid)
    root = OUT / "experiments"
    result = {"model": "kernel_baseline", "seconds": time.time() - started,
              **metrics(logits, reg, valid)}
    np.savez(root / "kernel_baseline_valid_predictions.npz", logits=logits, reg=reg)
    joblib.dump({"processors": processors, "classifier": classifier,
                 "regressor": regressor}, root / "kernel_baseline.joblib")
    save_json(root / "kernel_baseline_summary.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
