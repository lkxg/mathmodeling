import tempfile
import unittest
from pathlib import Path

import numpy as np

from q1.common import atomic_json, atomic_npz
from q1.dataset import collate
from q1.pipeline import Cache, aggregate, reviewed_quality, validate
from q1.temporal import (interval_pool, make_timeline,
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
        self.assertAlmostEqual(float(std[0, 0]), np.sqrt(128 / 9), places=6)
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
        atomic_npz(d / "audio.npz", acoustic=np.ones((2, 2)),
                   acoustic_intervals=[[0, .5], [.5, 1]], acoustic_valid=[not silent] * 2)
        atomic_json(d / "audio.json", {"acoustic_names": ["a", "b"]})
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
            # Word 0.2-0.7 overlaps both acoustic windows: 0.3 s and 0.2 s.
            offsets, indices = data["map_acoustic_offsets"], data["map_acoustic_indices"]
            self.assertEqual(offsets.tolist(), [0, 1, 3, 4])
            self.assertEqual(indices[offsets[1]:offsets[2]].tolist(), [0, 1])
            np.testing.assert_allclose(data["map_acoustic_weights"][offsets[1]:offsets[2]], [.6, .4], rtol=1e-6)
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

    def test_audio_export_is_25_means_and_25_within_interval_stds(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, cfg, sample = self._write_aggregate_inputs(tmp)
            x = np.arange(25)[None, :] + np.array([[3.], [8.]])
            atomic_npz(d / "audio.npz", acoustic=x, acoustic_intervals=[[0, .5], [.5, 1]],
                       acoustic_valid=[True, True])
            result = aggregate(sample, d, cfg)
            self.assertEqual(result["audio_dim"], 50)
            with np.load(Path(tmp) / "features/sample.npz") as z:
                np.testing.assert_allclose(z["audio"][1, :25], np.arange(25) + 5)
                np.testing.assert_allclose(z["audio"][1, 25:], np.sqrt(6), rtol=1e-6)
                self.assertEqual(z["coverage"].shape, (3, 2))

    def test_review_exclusion_removes_false_face_from_output_and_source_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            d, cfg, sample = self._write_aggregate_inputs(tmp)
            review = Path(tmp) / "review.json"
            atomic_json(review, {"cases": [{"key": sample["key"], "video_sha256": "test",
                                           "excluded_original_frames": [45, 48]}]})
            cfg["quality_review"] = str(review)
            atomic_npz(d / "vision.npz", features=np.ones((2, 2)), intervals=[[0, .5], [.5, 1]],
                       quality=[1, 1], frame_indices=[45, 48])
            result = aggregate(sample, d, cfg)
            self.assertEqual(result["review_excluded_vision_frames"], 2)
            with np.load(Path(tmp) / "features/sample.npz") as z:
                self.assertFalse(z["modality_mask"][:, 2].any())
                self.assertEqual(len(z["map_vision_indices"]), 0)
                self.assertTrue(z["source_vision_excluded"].all())
            with np.load(d / "vision.npz") as z:
                np.testing.assert_array_equal(z["quality"], [1, 1])

    def test_review_exclusion_rejects_different_source_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.json"
            atomic_json(path, {"cases": [{"key": "s", "video_sha256": "old",
                                         "excluded_original_frames": [0]}]})
            with self.assertRaisesRegex(ValueError, "哈希不匹配"):
                reviewed_quality({"key": "s", "video_sha256": "new"},
                                 {"quality": np.array([1]), "frame_indices": np.array([0])},
                                 {"quality_review": str(path)})

    def test_source_map_rejects_negative_index(self):
        from q1.validation import check_mapping
        z = {"timestamps": np.array([[0., 1.]]), "source_acoustic_intervals": np.array([[0., 1.]]),
             "map_acoustic_offsets": np.array([0, 1]), "map_acoustic_indices": np.array([-1]),
             "map_acoustic_weights": np.array([1.])}
        with self.assertRaisesRegex(ValueError, "索引或权重非法"):
            check_mapping(z, "acoustic", np.array([[0., 1.]]), np.ones((1, 25)), np.ones(1), "audio", 1)

    def test_real_opensmile_keeps_native_offset_and_marks_digital_silence(self):
        import soundfile as sf
        from q1.backends import AudioEncoder
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            encoder = AudioEncoder({})
            for silent in (False, True):
                wave = np.zeros(8000, np.float32) if silent else .1 * np.sin(2 * np.pi * 200 * np.arange(8000) / 16000)
                sf.write(directory / "audio.wav", wave, 16000, subtype="FLOAT")
                atomic_json(directory / "media.json", {"audio_offset": .2, "audio_rms": float(np.sqrt(np.mean(wave**2))),
                            "audio_peak": float(np.abs(wave).max()), "silent_audio": silent})
                encoder.process({}, directory, {})
                with np.load(directory / "audio.npz") as z:
                    self.assertEqual(z["acoustic"].shape[1], 25)
                    self.assertAlmostEqual(z["acoustic_intervals"][0, 0], .2)
                    self.assertEqual(z["acoustic_valid"].any(), not silent)

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
