import tempfile
import unittest
from pathlib import Path

import numpy as np

from q1.common import atomic_json, atomic_npz
from q1.dataset import collate
from q1.pipeline import Cache, aggregate, validate
from q1.temporal import (conv_geometry, conv_intervals, interval_pool, make_timeline,
                         pool_tokens, sampled_support, word_spans)


class TemporalTests(unittest.TestCase):
    def test_repeated_words_and_punctuation_preserve_original_offsets(self):
        raw = "It's good, good — high-end!"
        spans = word_spans(raw, ["It's", "good", "good", "highend"])
        self.assertEqual([raw[a:b] for a, b in spans], ["It's", "good", "good", "high-end"])
        with self.assertRaises(ValueError):
            word_spans(raw, ["It's", "good"])

    def test_special_tokens_are_not_pooled_and_subwords_are_combined(self):
        values = np.array([[99., 99.], [2., 4.], [4., 8.], [99., 99.]])
        x, mask, mapping = pool_tokens(values, [[0, 0], [0, 2], [2, 5], [0, 0]], [[0, 5]], [1, 1, 1, 0])
        np.testing.assert_array_equal(x, [[3, 6]])
        self.assertTrue(mask[0])
        self.assertEqual(mapping, [[1, 2]])

    def test_weighted_overlap_is_not_an_unweighted_mean(self):
        x, std, mask, coverage, mapping = interval_pool([[0, 2], [3, 4]],
            [[0, 1], [1, 3]], [[0], [8]], [1, .5])
        self.assertAlmostEqual(float(x[0, 0]), 8 / 3, places=6)
        self.assertAlmostEqual(sum(mapping[0]["weights"]), 1)
        np.testing.assert_array_equal(mask, [True, False])
        np.testing.assert_array_equal(coverage, [1, 0])
        self.assertEqual(x[1, 0], 0)

    def test_nan_and_zero_quality_do_not_contaminate_pooling(self):
        x, _, mask, coverage, mapping = interval_pool([[0, 2]], [[0, 1], [0, 2], [1, 2]],
                                                     [[np.nan], [5], [10]], [1, 0, 1])
        np.testing.assert_array_equal(x, [[10]])
        self.assertAlmostEqual(coverage[0], .5)
        self.assertEqual(mapping[0]["indices"], [2])

    def test_coverage_does_not_double_count_overlapping_receptive_fields(self):
        _, _, _, coverage, _ = interval_pool([[0, 1]], [[0, .7], [.3, 1]], [[1], [2]], [1, 1])
        self.assertEqual(coverage[0], 1)

    def test_convolution_anchor_uses_stride_and_offset_not_clip_stretching(self):
        rf, step = conv_geometry("[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2")
        self.assertEqual((rf, step), (400, 320))
        t = conv_intervals(49, 16000, 16000, .2, rf, step)
        np.testing.assert_allclose(t[0], [.2, .225])
        np.testing.assert_allclose(t[-1], [1.16, 1.185])
        with self.assertRaises(ValueError):
            conv_intervals(50, 16000, 16000, 0, rf, step)
        with self.assertRaises(ValueError):
            conv_geometry("__import__('os').system('false')")

    def test_gaps_are_retained_without_calling_them_silence(self):
        words = [{"text": "hi", "start": .2, "end": .6, "time_valid": True}]
        timeline = make_timeline(words, 1)
        self.assertEqual([r["kind"] for r in timeline], ["unassigned", "word", "unassigned"])
        self.assertEqual(timeline[0]["word_index"], -1)

    def test_equal_gaps_are_kept_regardless_of_float_rounding(self):
        # 3.28 - 3.2 < 0.08 and 2.0 - 1.92 > 0.08 in binary floating point.
        words = [{"text": t, "start": a, "end": b, "time_valid": True}
                 for t, a, b in [("a", 0, 1.92), ("b", 2.0, 3.2), ("c", 3.28, 4.48), ("d", 4.56, 5.0)]]
        gaps = [(r["start"], r["end"]) for r in make_timeline(words, 5.0, min_gap=.08)
                if r["kind"] == "unassigned"]
        self.assertEqual(gaps, [(1.92, 2.0), (3.2, 3.28), (4.48, 4.56)])
        self.assertEqual([r["kind"] for r in make_timeline(words, 5.07, min_gap=.08)][-1], "word")

    def test_sampled_video_support_preserves_offset(self):
        np.testing.assert_allclose(sampled_support([.2, .3, .5], .2, .6),
                                   [[.2, .25], [.25, .4], [.4, .6]])

    def test_cache_detects_corruption_and_configuration_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"output_dir": tmp, "setting": 1}
            sample = {"key": "s", "id": "s"}
            cache = Cache(cfg)
            cache.directory(sample).mkdir(parents=True)
            for p in cache.files(sample, "media"):
                p.write_bytes(b"original")
            cache.commit(sample, "media", cache.signature(sample, "media"), {})
            self.assertTrue(cache.valid(sample, "media"))
            changed = Cache({**cfg, "setting": 2})
            self.assertFalse(changed.valid(sample, "media"))
            cache.files(sample, "media")[0].write_bytes(b"corrupt")
            self.assertFalse(cache.valid(sample, "media"))

    def _write_aggregate_inputs(self, tmp, silent=False):
        d = Path(tmp) / "cache"
        cfg = {"output_dir": tmp, "aggregation": {"retain_gaps": True, "min_gap_seconds": .05,
                                                   "export_dtype": "float32", "alignment_suspect_ratio": .6}}
        sample = dict(key="sample", id="video$_$1", video_id="video", clip_id="1",
                      video_sha256="test", raw_text="good")
        atomic_json(d / "media.json", {"duration": 1., "origin_pts": .1, "silent_audio": silent})
        atomic_json(d / "align.json", {"words": [{"text": "good", "start": .2, "end": .7,
                                                   "time_valid": not silent, "char_span": [0, 4]}]})
        atomic_npz(d / "text.npz", features=np.ones((1, 4)), valid=np.array([True]))
        atomic_json(d / "text.json", {"word_to_tokens": [[1, 2]]})
        atomic_npz(d / "audio_features.npz", emotion=np.ones((2, 3)), emotion_intervals=[[0, .5], [.5, 1]],
                   emotion_valid=[not silent] * 2, acoustic=np.ones((2, 2)),
                   acoustic_intervals=[[0, .5], [.5, 1]], acoustic_valid=[not silent] * 2)
        atomic_json(d / "audio_features.json", {"acoustic_names": ["a", "b"]})
        atomic_npz(d / "vision.npz", features=np.zeros((2, 2)), intervals=[[0, .5], [.5, 1]], quality=[0, 0])
        atomic_json(d / "vision.json", {"frames": []})
        return d, cfg, sample

    def test_aggregate_preserves_missing_face_and_batch_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, cfg, sample = self._write_aggregate_inputs(tmp)
            summary = aggregate(sample, d, cfg)
            self.assertEqual(summary["positions"], 3)
            self.assertFalse(summary["alignment_suspect"])
            with np.load(Path(tmp) / "features/sample.npz", allow_pickle=False) as z:
                data = {k: z[k].copy() for k in z.files}
            np.testing.assert_array_equal(data["modality_mask"][:, 0], [False, True, False])
            self.assertFalse(data["modality_mask"][:, 2].any())
            self.assertTrue(data["valid_mask"].all())
            self.assertTrue((data["vision"] == 0).all())
            # Word 0.2-0.7 overlaps both emotion windows: 0.3 s and 0.2 s.
            offsets, indices = data["map_emotion_offsets"], data["map_emotion_indices"]
            self.assertEqual(offsets.tolist(), [0, 1, 3, 4])
            self.assertEqual(indices[offsets[1]:offsets[2]].tolist(), [0, 1])
            np.testing.assert_allclose(data["map_emotion_weights"][offsets[1]:offsets[2]], [.6, .4], rtol=1e-6)
            self.assertEqual(data["map_vision_offsets"].tolist(), [0, 0, 0, 0])
            short = {k: (v[1:2] if v.ndim else np.asarray(1, np.int32)) for k, v in data.items()}
            batch = collate([data, short])
            self.assertEqual(batch["text"].shape, (2, 3, 4))
            self.assertEqual(batch["valid_mask"][1].tolist(), [True, False, False])
            self.assertEqual(batch["modality_mask"][1, 1:].sum().item(), 0)

    def test_silent_audio_is_missing_and_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, cfg, sample = self._write_aggregate_inputs(tmp, silent=True)
            summary = aggregate(sample, d, cfg)
            self.assertEqual(summary["suspect_reasons"], "silent_audio;low_valid_alignment")
            self.assertEqual(summary["valid_audio_ratio"], 0)
            with np.load(Path(tmp) / "features/sample.npz", allow_pickle=False) as z:
                self.assertFalse(z["modality_mask"][:, 1].any())
                self.assertTrue((z["audio"] == 0).all())
                # Text survives even though no timestamp is usable.
                self.assertEqual(z["modality_mask"][:, 0].sum(), 1)

    def test_selected_validation_does_not_overwrite_full_manifest_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_file = Path(tmp) / "summary.csv"
            report_file.write_text("full report")
            result = validate({"output_dir": tmp, "expected_samples": 100}, [{"key": "s", "id": "s"}])
            self.assertEqual(result["scope"], "selected")
            self.assertEqual(result["complete"], 0)
            self.assertEqual(report_file.read_text(), "full report")
            self.assertTrue((Path(tmp) / "validation.selected.json").exists())


if __name__ == "__main__":
    unittest.main()
