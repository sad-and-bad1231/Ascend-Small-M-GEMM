import unittest

from decision_closure import headroom


class HeadroomTests(unittest.TestCase):
    def test_bytes_do_not_remove_required_outputs_or_weights(self):
        row = headroom(8, 40, .04)
        self.assertEqual(row["optimistic_avoidable_write_plus_read_bytes"], 81920)
        self.assertEqual(row["mandatory_final_q_k_v_output_bytes"], 40960)
        self.assertEqual(row["logical_weight_bytes_not_removed"], 26214400)

    def test_gap_only_is_not_consumer_erasure(self):
        row = headroom(1, 50, .02)
        self.assertAlmostEqual(row["scenario_gap_only_span_reduction_percent"], .04)
        self.assertAlmostEqual(row["scenario_additional_savings_needed_beyond_gap_us"], 2.48)
        self.assertTrue(row["profiled_median_scenario_is_not_formal_p95_or_service_gain"])

    def test_invalid_domain_not_silently_extrapolated(self):
        for values in ((16, 50, .02), (1, 0, 0), (1, 50, -1), (1, 1, 2)):
            with self.assertRaises(ValueError):
                headroom(*values)


if __name__ == "__main__":
    unittest.main()
