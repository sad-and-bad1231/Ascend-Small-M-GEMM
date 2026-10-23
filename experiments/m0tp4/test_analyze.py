import unittest
from pathlib import Path
from unittest.mock import patch

import analyze
import launch


class AnalysisTests(unittest.TestCase):
    def run_triplet(self, post=100, source_drift=False, output_drift=False):
        pairs = []
        for index, (arm, value) in enumerate((("B0", 100), ("B1", 98), ("B0", post))):
            window = {"median_TPOT_ms": value, "output_tokens_per_s": 10000 / value}
            pairs.append(({"arm": arm, "windows": {"w1": window, "w2": window}},
                          {"source": str(index) if source_drift else "frozen"}))
        records = [[1], [2] if output_drift else [1], [1]]
        with patch.object(analyze, "read_arm", side_effect=pairs), \
             patch.object(analyze, "outputs", side_effect=records):
            return analyze.analyze(Path("unused"), ["pre", "candidate", "post"])

    def test_valid_and_invalid_not_selected(self):
        self.assertAlmostEqual(self.run_triplet()["accepted_service_gain_percent"], 2)
        result = self.run_triplet(post=103)
        self.assertEqual(result["status"], "INVALID_ROLLBACK_DRIFT")
        self.assertIsNone(result["accepted_service_gain_percent"])
        self.assertGreater(result["raw_service_TPOT_gain_percent"], 2)

    def test_sources_and_full_output_must_match(self):
        for kwargs in ({"source_drift": True}, {"output_drift": True}):
            with self.assertRaises(AssertionError):
                self.run_triplet(**kwargs)

    def test_p95_nearest_rank(self):
        self.assertEqual(analyze.p95(list(range(1, 25))), 23)

    def test_terminal_hardware_stop_is_not_formal_permission(self):
        for permission in ({"hardware_stopped": True, "formal_approved": False},
                           {"hardware_stopped": False, "formal_approved": True}):
            with self.assertRaises(RuntimeError):
                launch.campaign_gate({"local_tp4_campaign_20261022": permission})

    def test_full_tokens_checked_even_if_hash_unchanged(self):
        reference = [{"output_token_ids": [1, 2], "prompt_token_ids": [3],
                      "payload_sha256": "same"} for _ in range(32)]
        candidate = [{**r, "output_token_ids": list(r["output_token_ids"])} for r in reference]
        candidate[0]["output_token_ids"][1] = 4
        with patch.object(analyze, "outputs", side_effect=[reference, candidate]), \
             patch.object(Path, "read_bytes", return_value=b"fixture"):
            report = analyze.token_report(Path("unused"), ["a", "b"])
            self.assertFalse(report["exact_match"])
            self.assertEqual(report["differences"][0]["first_differing_output_index"], 1)


if __name__ == "__main__":
    unittest.main()
