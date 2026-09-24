"""Grouped nested linear probes, with label-independent ridge operators.

One SVD per feature fold is reused for all alpha values and permutations. Every
permuted target still gets its own inner-CV hyperparameter selection and refit.
The operator includes the intercept and train-only imputation/standardization.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

CLASSES = np.array([-1, 0, 1])


def nested_splits(groups, inner_folds=5):
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 3:
        raise ValueError("嵌套分组验证至少需要3个原视频")
    result = []
    for train, test in LeaveOneGroupOut().split(groups, groups=groups):
        inner = GroupKFold(n_splits=min(inner_folds, len(np.unique(groups[train]))))
        folds = [(train[a], train[b]) for a, b in inner.split(train, groups=groups[train])]
        result.append((train, test, folds))
    return result


def ridge_operators(x, train, test, weights, alphas):
    """Return [alpha, test row, train row] maps; no targets enter this function."""
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    scaler = StandardScaler()
    a = scaler.fit_transform(imputer.fit_transform(x[train])) * weights
    b = scaler.transform(imputer.transform(x[test])) * weights
    center = a.mean(0)
    a, b = a - center, b - center
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    shrink = singular[None, :] / (singular[None, :] ** 2 + np.asarray(alphas)[:, None])
    h = np.einsum("ik,ak,jk->aij", b @ vt.T, shrink, u, optimize=True)
    # y_hat = H(y - mean(y)) + mean(y). Keep the intercept unpenalized.
    return h - h.mean(axis=2, keepdims=True) + 1 / len(train)


def within_group_targets(y, groups, permutations, rng):
    """Column 0 is observed; remaining columns permute labels within video_id.

This tests within-video exchangeability, conditional on each video's observed
label multiset. It does not test the removal of between-video associations.
"""
    y, groups = np.asarray(y), np.asarray(groups)
    result = np.repeat(y[:, None], permutations + 1, axis=1)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        for k in range(1, permutations + 1):
            result[idx, k] = rng.permutation(y[idx])
    return result


def class_counts(truth, prediction):
    """Counts [alpha, permutation, class, (TP, predicted, actual)]."""
    rows = []
    for c in CLASSES:
        actual, predicted = truth[None, :, :] == c, prediction == c
        rows.append(np.stack([np.sum(actual & predicted, axis=1),
                              np.sum(predicted, axis=1),
                              np.broadcast_to(np.sum(actual, axis=1), predicted.shape[::2])], axis=-1))
    return np.stack(rows, axis=-2)


def macro_from_counts(counts):
    denom = counts[..., 1] + counts[..., 2]
    return np.divide(2 * counts[..., 0], denom, out=np.zeros_like(denom, dtype=float),
                     where=denom > 0).mean(axis=-1)


def _predict(h, train, targets, labels):
    p = targets.shape[1]
    onehot = labels[train, :, None] == CLASSES
    joined = np.concatenate([targets[train], onehot.reshape(len(train), -1)], axis=1)
    fitted = (h.reshape(-1, len(train)) @ joined).reshape(*h.shape[:2], -1)
    regression = fitted[..., :p]
    scores = fitted[..., p:].reshape(*h.shape[:2], p, len(CLASSES))
    # Do not predict a class absent from a particular training fold.
    present = onehot.any(axis=0)
    scores = np.where(present[None, None, :, :], scores, -np.inf)
    classification = CLASSES[scores.argmax(axis=-1)]
    return regression, classification


def nested_predict(x, targets, groups, weights, alphas, inner_folds=5, progress=None):
    """Ridge regression + one-vs-rest ridge classification, separately tuned.

Regression alpha minimizes pooled inner held-out MAE; classification alpha
maximizes pooled inner held-out macro-F1 over the fixed three classes. Ties
choose the larger penalty. The outer held-out group's targets are never used.
"""
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[:, None]
    labels = np.sign(targets).astype(np.int8)
    alphas = np.asarray(alphas)
    regression = np.empty_like(targets)
    classification = np.empty(targets.shape, dtype=np.int8)
    folds = []
    splits = nested_splits(groups, inner_folds)
    for fold, (train, test, inner) in enumerate(splits):
        errors = np.zeros((len(alphas), targets.shape[1]))
        counts = np.zeros((len(alphas), targets.shape[1], len(CLASSES), 3))
        for fit, valid in inner:
            h = ridge_operators(x, fit, valid, weights, alphas)
            pr, pc = _predict(h, fit, targets, labels)
            errors += np.abs(pr - targets[valid][None, :, :]).sum(axis=1)
            counts += class_counts(labels[valid], pc)
        f1 = macro_from_counts(counts)
        # alphas is increasing; reversing implements a deterministic large-alpha tie break.
        best_r = len(alphas) - 1 - errors[::-1].argmin(axis=0)
        best_c = len(alphas) - 1 - f1[::-1].argmax(axis=0)
        h = ridge_operators(x, train, test, weights, alphas)
        pr, pc = _predict(h, train, targets, labels)
        columns = np.arange(targets.shape[1])
        regression[test] = pr[best_r, :, columns].T
        classification[test] = pc[best_c, :, columns].T
        folds.append({"test_group": str(np.asarray(groups)[test[0]]),
                      "train_indices": train.tolist(), "test_indices": test.tolist(),
                      "inner_folds": [{"train_indices": a.tolist(), "valid_indices": b.tolist()}
                                      for a, b in inner],
                      "regression_alpha": float(alphas[best_r[0]]),
                      "classification_alpha": float(alphas[best_c[0]])})
        if progress:
            progress(fold + 1, len(splits))
    return regression, classification, folds


def batch_metrics(y, pred, classes):
    """One score per column; all 100 rows, including neutral, enter primary scores."""
    y, pred, classes = (np.asarray(a) for a in (y, pred, classes))
    if y.ndim == 1:
        y, pred, classes = y[:, None], pred[:, None], classes[:, None]
    yc, pc = y - y.mean(axis=0), pred - pred.mean(axis=0)
    denom = np.sqrt((yc ** 2).sum(axis=0) * (pc ** 2).sum(axis=0))
    corr = np.divide((yc * pc).sum(axis=0), denom,
                     out=np.full(y.shape[1], np.nan), where=denom > 1e-14)
    labels = np.sign(y).astype(np.int8)
    counts = class_counts(labels, classes[None, :, :])[0]
    nz = y != 0
    return {"mae": np.mean(np.abs(y - pred), axis=0), "pearson": corr,
            "accuracy3": np.mean(classes == labels, axis=0),
            "macro_f1_3": macro_from_counts(counts),
            "acc2_regression_sign": np.divide((((pred > 0) == (y > 0)) & nz).sum(axis=0),
                                             nz.sum(axis=0), out=np.full(y.shape[1], np.nan),
                                             where=nz.sum(axis=0) > 0)}


def cluster_bootstrap_indices(groups, repeats, rng):
    members = [np.flatnonzero(np.asarray(groups) == g) for g in np.unique(groups)]
    return [np.concatenate([members[k] for k in draw])
            for draw in rng.integers(0, len(members), (repeats, len(members)))]


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(pvalues[order] *
                                                        np.arange(len(order), 0, -1)))
    return adjusted
