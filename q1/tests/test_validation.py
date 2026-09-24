import unittest

from q1.validation import invalid_reason, temporal_support, validation_metrics


class CoverageTests(unittest.TestCase):
    def test_timeline_can_be_complete_without_any_valid_source(self):
        metrics = temporal_support([[0, 3], [2, 10]], [], [])
        self.assertEqual(metrics["timeline_seconds"], 10)
        self.assertTrue(all(metrics[k] == 0 for k in metrics if k != "timeline_seconds"))

    def test_clipped_union_and_intersection_do_not_double_count(self):
        # I=[0,4]∪[6,10]; source supports extend outside I and overlap internally.
        metrics = temporal_support([[0, 4], [6, 10]], [[-1, 3], [2, 8]], [[1, 7], [9, 12]])
        self.assertEqual(metrics, {"timeline_seconds": 8, "audio_support_seconds": 6,
                                  "vision_support_seconds": 5, "union_support_seconds": 7,
                                  "intersection_support_seconds": 4})
        self.assertEqual(metrics["union_support_seconds"], metrics["audio_support_seconds"] + metrics["vision_support_seconds"] - metrics["intersection_support_seconds"])

    def test_dataset_rates_use_seconds_instead_of_mean_clip_percentages(self):
        rows = [{"status": "ok", "timeline_seconds": 1, "duration": 1, "audio_support_seconds": 1, "invalid_words": 0},
                {"status": "ok", "timeline_seconds": 9, "duration": 10, "audio_support_seconds": 0, "invalid_words": 0}]
        result = validation_metrics(rows)
        self.assertEqual(result["temporal_coverage"]["audio"]["ratio"], .1)
        self.assertEqual(result["temporal_coverage"]["timeline_retention"]["ratio"], 10 / 11)

    def test_invalid_reasons_have_explicit_exclusive_priority(self):
        self.assertEqual(invalid_reason({"flags": ["non_positive_duration", "silent_audio", "out_of_audio_bounds"]}), "silent_audio")
        self.assertEqual(invalid_reason({"flags": ["non_positive_duration", "out_of_audio_bounds"]}), "out_of_audio_bounds")
        self.assertEqual(invalid_reason({"flags": ["unrecognized_reason"]}), "other")


if __name__ == "__main__":
    unittest.main()
