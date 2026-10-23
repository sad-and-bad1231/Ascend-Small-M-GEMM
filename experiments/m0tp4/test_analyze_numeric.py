import unittest

import torch

from analyze_numeric import check_inputs, compare_logits, compare_tensor


class NumericTests(unittest.TestCase):
    def test_logits_near_tie_is_not_exact_pass(self):
        a = torch.tensor([[1.0, 1.01, 0.0]])
        b = torch.tensor([[1.02, 1.01, 0.0]])
        result = compare_logits(a, b)
        self.assertFalse(result["equal"])
        self.assertEqual(result["top1_agreement"], 0)
        self.assertNotIn("top1_ids", result)

    def test_same_logits_and_shape_mismatch(self):
        a = torch.zeros((2, 3), dtype=torch.bfloat16)
        self.assertTrue(compare_tensor(a, a)["equal"])
        self.assertFalse(compare_tensor(a, a[:, :2])["equal"])

    def test_actual_input_prefix_contract(self):
        frozen = {"prompt_ids": [[i] for i in range(8)],
                  "forced_prefix": [[i+10, i+20, i+30] for i in range(8)]}
        steps = [{"metadata": {"scheduled_tokens": [1]*8}, "input_ids": torch.arange(8)}]
        steps += [{"input_ids": torch.arange(8)+offset} for offset in (10, 20, 30)]
        self.assertTrue(all(check_inputs(steps, frozen).values()))
        steps[1]["input_ids"][0] = -1
        self.assertFalse(check_inputs(steps, frozen)["actual_decode_inputs_match_forced_prefix"])


if __name__ == "__main__":
    unittest.main()
