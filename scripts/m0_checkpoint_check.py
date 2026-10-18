#!/usr/bin/env python3
"""CPU-only checkpoint inventory; does not import frameworks or hash tensor data."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def inspect(root):
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())
    assert config["model_type"] == "qwen3"
    assert (config["hidden_size"], config["num_attention_heads"],
            config["num_key_value_heads"], config["head_dim"]) == (5120, 64, 8, 128)
    assert config["torch_dtype"] == "bfloat16"
    shards = sorted(set(index["weight_map"].values()))
    seen, payload, shard_bytes = set(), 0, {}
    for name in shards:
        path = root / name
        assert path.parent == root and path.is_file(), name
        size = path.stat().st_size
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            assert 0 < header_len <= 16 * 1024 * 1024
            header = json.loads(handle.read(header_len))
        spans = []
        for tensor, metadata in header.items():
            if tensor == "__metadata__":
                continue
            assert tensor not in seen and index["weight_map"][tensor] == name
            assert metadata["dtype"] == "BF16", tensor
            start, end = metadata["data_offsets"]
            elements = 1
            for dimension in metadata["shape"]:
                elements *= dimension
            assert end - start == elements * 2
            spans.append((start, end))
            seen.add(tensor)
        cursor = 0
        for start, end in sorted(spans):
            assert start == cursor and end >= start
            cursor = end
        assert 8 + header_len + cursor == size
        payload += cursor
        shard_bytes[name] = size
    assert seen == set(index["weight_map"])
    assert payload == index["metadata"]["total_size"]
    metadata_hashes = {}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json",
                 "tokenizer_config.json", "generation_config.json", "merges.txt", "vocab.json"):
        metadata_hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return {"status": "PASS_METADATA_AND_HEADER_ONLY", "config": config,
            "shard_count": len(shards), "tensor_count": len(seen),
            "payload_bytes": payload, "shard_bytes": shard_bytes,
            "metadata_sha256": metadata_hashes,
            "weight_content_hash": "NOT_COMPUTED", "historical_identity": "UNKNOWN",
            "npu_loaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.checkpoint.resolve()), indent=2))
