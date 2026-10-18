import json
import struct
import tempfile
import unittest
from pathlib import Path

from m0_checkpoint_check import inspect


class CheckpointTests(unittest.TestCase):
    def make_checkpoint(self, root):
        config = {"model_type": "qwen3", "hidden_size": 5120,
                  "num_attention_heads": 64, "num_key_value_heads": 8,
                  "head_dim": 128, "torch_dtype": "bfloat16"}
        (root / "config.json").write_text(json.dumps(config))
        shard = "model-00001-of-00001.safetensors"
        header = json.dumps({"w": {"dtype": "BF16", "shape": [2],
                                  "data_offsets": [0, 4]}}).encode()
        (root / shard).write_bytes(struct.pack("<Q", len(header)) + header + bytes(4))
        index = {"metadata": {"total_size": 4}, "weight_map": {"w": shard}}
        (root / "model.safetensors.index.json").write_text(json.dumps(index))
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                     "merges.txt", "vocab.json"):
            (root / name).write_text("{}")
        return root / shard

    def test_valid_inventory_is_not_content_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root)
            result = inspect(root)
            self.assertEqual(result["tensor_count"], 1)
            self.assertEqual(result["weight_content_hash"], "NOT_COMPUTED")
            self.assertFalse(result["npu_loaded"])

    def test_missing_shard_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root).unlink()
            with self.assertRaises(AssertionError):
                inspect(root)

    def test_truncated_payload_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = self.make_checkpoint(root)
            shard.write_bytes(shard.read_bytes()[:-1])
            with self.assertRaises(AssertionError):
                inspect(root)


if __name__ == "__main__":
    unittest.main()
