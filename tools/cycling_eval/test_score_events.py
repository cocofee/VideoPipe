import json
import math
import tempfile
import unittest
from pathlib import Path

from score_events import load_events, match_events, score


class ScoreEventsTests(unittest.TestCase):
    def test_exact_match(self):
        ground_truth = [{"athlete_id": "a", "timestamp_ms": 1000, "ocr": "42"}]
        predictions = [{"athlete_id": "a", "timestamp_ms": 1000, "ocr": "42"}]
        self.assertEqual(match_events(ground_truth, predictions, 1000), [(0, 0)])
        result = score(ground_truth, predictions)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["false_event_rate"], 0.0)
        self.assertEqual(result["ocr_confirmed_rate"], 1.0)

    def test_timestamp_tolerance(self):
        ground_truth = [{"athlete_id": "a", "timestamp_ms": 1000}]
        predictions = [{"athlete_id": "a", "timestamp_ms": 1500}]
        self.assertEqual(len(match_events(ground_truth, predictions, 500)), 1)
        self.assertEqual(len(match_events(ground_truth, predictions, 499)), 0)

    def test_matching_is_one_to_one(self):
        ground_truth = [{"athlete_id": "a", "timestamp_ms": 1000}]
        predictions = [
            {"athlete_id": "a", "timestamp_ms": 1000},
            {"athlete_id": "a", "timestamp_ms": 1001},
        ]
        self.assertEqual(len(match_events(ground_truth, predictions, 1000)), 1)
        result = score(ground_truth, predictions)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["duplicate_rate"], 0.5)
        self.assertEqual(result["false_event_rate"], 0.0)

    def test_ocr_mismatch_keeps_athlete_match(self):
        ground_truth = [{"athlete_id": "a", "timestamp_ms": 1000, "ocr": "42"}]
        predictions = [{"athlete_id": "a", "timestamp_ms": 1000, "ocr": "24"}]
        result = score(ground_truth, predictions)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["ocr_confirmed_rate"], 0.0)

    def test_missing_evidence(self):
        ground_truth = [{"athlete_id": "a", "timestamp_ms": 1000}]
        predictions = [{"athlete_id": "a", "timestamp_ms": 1000, "evidence": {}}]
        result = score(ground_truth, predictions)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["evidence_complete_rate"], 0.0)

    def test_rejects_missing_athlete_id_in_all_inputs(self):
        valid_event = {"athlete_id": "a", "timestamp_ms": 1000}
        invalid_event = {"timestamp_ms": 1000}
        for ground_truth, predictions in (
            ([invalid_event], [valid_event]),
            ([valid_event], [invalid_event]),
        ):
            with self.subTest(ground_truth=ground_truth, predictions=predictions):
                with self.assertRaisesRegex(ValueError, "athlete_id"):
                    score(ground_truth, predictions)

    def test_rejects_missing_timestamp_in_all_inputs(self):
        valid_event = {"athlete_id": "a", "timestamp_ms": 1000}
        invalid_event = {"athlete_id": "a"}
        for ground_truth, predictions in (
            ([invalid_event], [valid_event]),
            ([valid_event], [invalid_event]),
        ):
            with self.subTest(ground_truth=ground_truth, predictions=predictions):
                with self.assertRaisesRegex(ValueError, "timestamp_ms"):
                    score(ground_truth, predictions)

    def test_rejects_non_finite_and_invalid_timestamp_types(self):
        valid_event = {"athlete_id": "a", "timestamp_ms": 1000}
        for invalid_timestamp in (math.nan, math.inf, -math.inf, "1000", True, []):
            invalid_event = {"athlete_id": "a", "timestamp_ms": invalid_timestamp}
            for ground_truth, predictions in (
                ([invalid_event], [valid_event]),
                ([valid_event], [invalid_event]),
            ):
                with self.subTest(
                    timestamp=invalid_timestamp,
                    ground_truth=ground_truth,
                    predictions=predictions,
                ):
                    with self.assertRaisesRegex(ValueError, "timestamp_ms"):
                        score(ground_truth, predictions)

    def test_rejects_integer_timestamp_too_large_for_float(self):
        valid_event = {"athlete_id": "a", "timestamp_ms": 1000}
        invalid_event = {"athlete_id": "a", "timestamp_ms": 10**10000}
        for label, ground_truth, predictions in (
            ("ground truth", [invalid_event], [valid_event]),
            ("prediction", [valid_event], [invalid_event]),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"^{label} event at index 0: each event requires finite timestamp_ms$",
                ):
                    score(ground_truth, predictions)

    def test_evidence_paths_must_be_non_empty_strings(self):
        ground_truth = [{"athlete_id": "a", "timestamp_ms": 1000}]
        invalid_evidence_values = (
            {"frame_path": 0, "clip_path": "clip.mp4"},
            {"frame_paths": [0, "frame.jpg"], "clip_path": "clip.mp4"},
            {"frame_paths": [], "clip_path": "clip.mp4"},
            {"frame_path": "frame.jpg", "clip_path": []},
            {"frame_path": {}, "clip_path": "clip.mp4"},
            {"frame_path": "frame.jpg", "clip_path": 1},
        )
        for evidence in invalid_evidence_values:
            with self.subTest(evidence=evidence):
                predictions = [
                    {
                        "athlete_id": "a",
                        "timestamp_ms": 1000,
                        "evidence": evidence,
                    }
                ]
                self.assertEqual(
                    score(ground_truth, predictions)["evidence_complete_rate"],
                    0.0,
                )

    def test_evidence_accepts_non_empty_frame_paths(self):
        ground_truth = [{"athlete_id": "a", "timestamp_ms": 1000}]
        predictions = [
            {
                "athlete_id": "a",
                "timestamp_ms": 1000,
                "evidence": {
                    "frame_paths": ["frame-1.jpg", "frame-2.jpg"],
                    "clip_path": "clip.mp4",
                },
            }
        ]
        self.assertEqual(
            score(ground_truth, predictions)["evidence_complete_rate"],
            1.0,
        )

    def test_load_events_accepts_wrapped_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text(json.dumps({"events": [{"athlete_id": "a", "timestamp_ms": 1}]}), encoding="utf-8")
            self.assertEqual(load_events(path), [{"athlete_id": "a", "timestamp_ms": 1}])


if __name__ == "__main__":
    unittest.main()
