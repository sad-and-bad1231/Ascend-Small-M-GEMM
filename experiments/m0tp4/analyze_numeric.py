#!/usr/bin/env python3
"""CPU-only comparison of private post-forward receipts; never exports token IDs."""
import argparse
import json
from pathlib import Path

import torch


def compare_tensor(a, b):
    if a is None or b is None:
        return {"available": False}
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"available": True, "same_shape_dtype": False, "equal": False}
    delta = (a.float() - b.float()).abs()
    return {"available": True, "same_shape_dtype": True, "shape": list(a.shape),
            "dtype": str(a.dtype), "equal": bool(torch.equal(a, b)),
            "different_elements": int((a != b).sum()),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "max_abs": float(delta.max()), "mean_abs": float(delta.mean())}


def compare_logits(a, b):
    result = compare_tensor(a, b)
    av = a.float().topk(2, dim=-1).values
    bv = b.float().topk(2, dim=-1).values
    result.update(top1_agreement=int((a.argmax(-1) == b.argmax(-1)).sum()), rows=a.shape[0],
                  top1_top2_margin_a=(av[:, 0] - av[:, 1]).tolist(),
                  top1_top2_margin_b=(bv[:, 0] - bv[:, 1]).tolist())
    return result


def check_inputs(steps, frozen):
    first = steps[0]
    sizes = first["metadata"]["scheduled_tokens"]
    prompts = frozen["prompt_ids"]
    expected = [token for prompt in prompts for token in prompt]
    prompt_ok = sizes == [len(x) for x in prompts] and torch.equal(
        first["input_ids"][:len(expected)], torch.tensor(expected))
    prefix_ok = all(torch.equal(steps[step]["input_ids"][:8],
        torch.tensor([x[step-1] for x in frozen["forced_prefix"]])) for step in range(1, 4))
    return {"prompt_ids_in_frozen_order": prompt_ok,
            "actual_decode_inputs_match_forced_prefix": prefix_ok}


def analyze(left, right):
    records = []
    left_ranks, right_ranks = [], []
    for rank in range(4):
        a = torch.load(left / f"rank{rank}_numeric.local.pt", map_location="cpu", weights_only=True)
        b = torch.load(right / f"rank{rank}_numeric.local.pt", map_location="cpu", weights_only=True)
        assert len(a) == len(b) and len(a) >= 4
        left_ranks.append(a)
        right_ranks.append(b)
        for index, (x, y) in enumerate(zip(a, b)):
            item = {"rank": rank, "step": index,
                    "metadata_a": x["metadata"], "metadata_b": y["metadata"],
                    "metadata_equal": x["metadata"] == y["metadata"]}
            for name in ("input_ids", "positions", "hidden", "local_logits"):
                item[name] = compare_tensor(x[name], y[name])
            item["logits"] = compare_logits(x["logits"], y["logits"])
            records.append(item)
    cross_rank = []
    for label, ranks in (("a", left_ranks), ("b", right_ranks)):
        for rank in range(1, 4):
            for step in range(len(ranks[0])):
                cross_rank.append({"side": label, "rank": rank, "step": step,
                    "hidden": compare_tensor(ranks[0][step]["hidden"], ranks[rank][step]["hidden"]),
                    "logits": compare_tensor(ranks[0][step]["logits"], ranks[rank][step]["logits"])})
    identity_a = json.loads((left / "identity.local.json").read_text())
    identity_b = json.loads((right / "identity.local.json").read_text())
    frozen = json.loads((left.parent / "frozen_inputs.local.json").read_text())
    return {"formal_approved": False, "performance_measurement": False,
            "teacher_forced_tokens_are_not_correctness_evidence": True,
            "same_frozen_input_hash": identity_a["input_sha256"] == identity_b["input_sha256"],
            "same_config": identity_a["config"] == identity_b["config"],
            "input_contract_a": [check_inputs(x, frozen) for x in left_ranks],
            "input_contract_b": [check_inputs(x, frozen) for x in right_ranks],
            "device_inputs_equal": all(x["input_ids"]["equal"] and x["positions"]["equal"] for x in records),
            "raw_logits_equal": all(x["logits"]["equal"] for x in records),
            "raw_top1_equal": all(x["logits"]["top1_agreement"] == x["logits"]["rows"] for x in records),
            "records": records, "within_process_cross_rank": cross_rank}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze(args.left, args.right)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "within_process_cross_rank")}))
