import unittest

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from q1.analysis.probe import A793, A818, aligned_blocks, design
from q1.analysis.probe_cv import (CLASSES, batch_metrics, cluster_bootstrap_indices,
                                  holm, nested_predict, nested_splits, ridge_operators,
                                  within_group_targets)


def conventional(weights, alpha, classification=False):
    head = RidgeClassifier(alpha=alpha, solver="svd") if classification else Ridge(alpha=alpha, solver="svd")
    return make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                         FunctionTransformer(lambda x: x * weights), head)


class ProbeTests(unittest.TestCase):
    def test_exported_within_word_std_is_preserved_and_invalid_rows_excluded(self):
        z = {"audio": np.full((4, 818), 999.), "vision": np.full((4, 56), 999.),
             "text": np.ones((4, 768)), "word_indices": np.array([0, 1, 2, -1]),
             "time_valid_mask": np.array([True, True, False, True]),
             "component_mask": np.ones((4, 4), bool)}
        z["audio"][:2, 768:793] = 7  # Between-word std is zero.
        z["audio"][0, 793:] = 3
        z["audio"][1, 793:] = 5
        z["vision"][0, 28:] = 2
        z["vision"][1, 28:] = 6
        block = aligned_blocks(z)
        np.testing.assert_array_equal(block["lld_std_aligned"], np.full(25, 4.))
        np.testing.assert_array_equal(block["vision_std_aligned"], np.full(28, 4.))
        z["component_mask"][:, 2] = False
        self.assertTrue(np.isnan(aligned_blocks(z)["lld_std_aligned"]).all())

    def test_ablation_keeps_existing_coordinate_weights(self):
        blocks = {"e2v_aligned": np.ones((3, 768)), "lld_mean_aligned": np.ones((3, 25)),
                  "lld_std_aligned": np.ones((3, 25))}
        a, w = design(blocks, A793)
        b, v = design(blocks, A818)
        self.assertEqual((a.shape[1], b.shape[1]), (793, 818))
        np.testing.assert_array_equal(w, v[:793])

    def test_ridge_operator_matches_sklearn_with_nan_and_test_only_extremes(self):
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(12, 17)), rng.normal(size=(12, 4))
        x[:8, 0] = np.nan  # Completely empty training column.
        x[3, 2] = np.nan
        x[8:, 0] = 1e6
        x[8:, 4] *= 100
        train, test = np.arange(8), np.arange(8, 12)
        weights = np.linspace(.05, .8, 17)
        for alpha, h in zip([.01, 1., 100.], ridge_operators(x, train, test, weights, [.01, 1., 100.])):
            expected = conventional(weights, alpha).fit(x[train], y[train]).predict(x[test])
            np.testing.assert_allclose(h @ y[train], expected, rtol=1e-9, atol=1e-8)

    def test_nested_predictions_match_independent_sklearn_refits_for_every_permutation(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(18, 8))
        x[1, 0] = np.nan
        groups = np.repeat(np.arange(6), 3)
        y = np.tile([-1., 0., 1.], 6) * rng.uniform(.5, 2., 18)
        targets = within_group_targets(y, groups, 2, rng)
        alphas, weights = np.array([.1, 1., 10.]), np.linspace(.1, .5, 8)
        actual_r, actual_c, _ = nested_predict(x, targets, groups, weights, alphas, 3)
        for train, test, inner in nested_splits(groups, 3):
            self.assertFalse(set(groups[train]) & set(groups[test]))
            for fit, valid in inner:
                self.assertFalse(set(groups[fit]) & set(groups[valid]))
                self.assertFalse(set(test) & set(fit))
                self.assertFalse(set(test) & set(valid))
            for k in range(targets.shape[1]):
                yy, labels = targets[:, k], np.sign(targets[:, k])
                scores_r, scores_c = [], []
                for alpha in alphas:
                    errors, truth, predictions = [], [], []
                    for fit, valid in inner:
                        rr = conventional(weights, alpha).fit(x[fit], yy[fit]).predict(x[valid])
                        cc = conventional(weights, alpha, True).fit(x[fit], labels[fit]).predict(x[valid])
                        errors.extend(abs(rr - yy[valid]))
                        truth.extend(labels[valid]); predictions.extend(cc)
                    scores_r.append(np.sum(errors))
                    scores_c.append(f1_score(truth, predictions, labels=CLASSES, average="macro", zero_division=0))
                ar = alphas[len(alphas) - 1 - np.argmin(scores_r[::-1])]
                ac = alphas[len(alphas) - 1 - np.argmax(scores_c[::-1])]
                expected_r = conventional(weights, ar).fit(x[train], yy[train]).predict(x[test])
                expected_c = conventional(weights, ac, True).fit(x[train], labels[train]).predict(x[test])
                np.testing.assert_allclose(actual_r[test, k], expected_r, atol=1e-10)
                np.testing.assert_array_equal(actual_c[test, k], expected_c)

    def test_group_permutation_preserves_video_label_multisets(self):
        y, groups = np.array([-1., 0., 2., 3., 8.]), np.array([0, 0, 0, 1, 2])
        p = within_group_targets(y, groups, 30, np.random.default_rng(3))
        np.testing.assert_array_equal(p[:, 0], y)
        for g in np.unique(groups):
            for k in range(p.shape[1]):
                np.testing.assert_array_equal(np.sort(p[groups == g, k]), np.sort(y[groups == g]))
        self.assertTrue(np.any(p[:3, 1:] != y[:3, None]))

    def test_cluster_bootstrap_keeps_every_clip_of_selected_video(self):
        groups = np.array([0, 0, 0, 1, 1, 2])
        for draw in cluster_bootstrap_indices(groups, 20, np.random.default_rng(4)):
            counts = np.bincount(draw, minlength=len(groups))
            for g in np.unique(groups):
                self.assertEqual(len(set(counts[groups == g])), 1)

    def test_classification_includes_neutral_and_matches_sklearn(self):
        y = np.array([-1., 0., 0., 2., 1.])
        r = np.array([.5, -1., 0., 1., -1.])
        c = np.array([0, 0, 1, 1, -1])
        m = batch_metrics(y, r, c)
        self.assertAlmostEqual(m["accuracy3"][0], accuracy_score(np.sign(y), c))
        self.assertAlmostEqual(m["macro_f1_3"][0], f1_score(np.sign(y), c, labels=CLASSES, average="macro"))
        self.assertAlmostEqual(m["acc2_regression_sign"][0], 1 / 3)

    def test_holm_adjustment(self):
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])


if __name__ == "__main__":
    unittest.main()
