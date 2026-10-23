import unittest

import torch

from analyze_numeric import compare_logits, compare_tensor


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


if __name__ == "__main__":
    unittest.main()
