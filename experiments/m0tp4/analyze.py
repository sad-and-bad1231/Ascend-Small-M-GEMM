#!/usr/bin/env python3
"""Sanitize one declared TP4 triplet; never promote invalid gains."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def read_arm(raw, label):
    path = raw / label
    launch = json.loads((raw / f"{label}.launch.local.json").read_text())
    assert launch["exit_code"] == 0 and launch["completed"]
    inventory = json.loads((path / "inventory.local.json").read_text())
    assert not inventory["profile"], "profiled latency is not performance evidence"
    windows = {}
    for window in (1, 2):
        rows = [json.loads(line) for line in (path / f"w{window}_requests.jsonl").read_text().splitlines()]
        aggregate = json.loads((path / f"w{window}_aggregate.json").read_text())
        assert len(rows) == (8 if window == 1 else 24)
        assert all(r["finished"] and r["output_tokens"] == 64 for r in rows)
        windows[f"w{window}"] = {
            "requests": len(rows), "output_tokens": len(rows) * 64,
            "median_TPOT_ms": statistics.median(r["tpot_ms"] for r in rows),
            "p95_TPOT_ms": p95([r["tpot_ms"] for r in rows]),
            "median_TTFT_ms": statistics.median(r["ttft_ms"] for r in rows),
            "p95_TTFT_ms": p95([r["ttft_ms"] for r in rows]),
            "p95_latency_ms": p95([r["request_latency_ms"] for r in rows]),
            "output_tokens_per_s": aggregate["output_tokens_per_s"],
            "per_request": [{key: r[key] for key in ("request_idx", "prompt_tokens", "output_tokens",
                             "ttft_ms", "tpot_ms", "request_latency_ms")} for r in rows]}
    policies = [json.loads((path / f"rank{i}_policy.local.json").read_text()) for i in range(4)]
    result = {"label": label, "arm": inventory["arm"], "windows": windows,
              "load_and_graph_init_s": inventory["load_and_graph_init_s"],
              "memory_per_rank": inventory["memory"],
              "converted_weights_per_rank": [p["converted"] for p in policies],
              "conversion_s_per_rank": [p["conversion_wall_s"] for p in policies],
              "raw_output_sha256": inventory["raw_output_sha256"]}
    return result, launch["source_sha256"]


def outputs(raw, label):
    return [json.loads(line) for window in (1, 2) for line in
            (raw / label / f"w{window}_outputs.local.jsonl").read_text().splitlines()]


def token_report(raw, labels):
    reference, candidate = [outputs(raw, label) for label in labels]
    assert len(reference) == len(candidate) == 32
    differences = []
    for index, (a, b) in enumerate(zip(reference, candidate)):
        fields = [key for key in a if a[key] != b[key]]
        pairs = list(zip(a["output_token_ids"], b["output_token_ids"]))
        if fields:
            differences.append({"request_index": index, "different_fields": fields,
                "same_prompt_ids": a["prompt_token_ids"] == b["prompt_token_ids"],
                "first_differing_output_index": next((i for i, (x, y) in enumerate(pairs) if x != y), None),
                "differing_output_positions": sum(x != y for x, y in pairs)})
    return {"labels": labels, "records": 32, "differing_requests": len(differences),
            "exact_match": reference == candidate, "differences": differences,
            "raw_output_sha256": {label: {f"w{w}": hashlib.sha256(
                (raw / label / f"w{w}_outputs.local.jsonl").read_bytes()).hexdigest()
                for w in (1, 2)} for label in labels}}


def campaign_summary(raw):
    processes = []
    for path in sorted(raw.glob("*.launch.local.json")):
        label = path.name.removesuffix(".launch.local.json")
        launch = json.loads(path.read_text())
        identity_path = raw / label / "identity.local.json"
        identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
        record = {"label": label, "exit_code": launch["exit_code"],
                  "complete_receipt": launch["completed"],
                  "released": launch["hbm_processes_released"], "process_wall_s": launch["wall_s"],
                  "queue_barrier": identity.get("queue_barrier", False),
                  "arm": identity.get("arm"), "profile": identity.get("profile"),
                  "source_sha256": {Path(key).name: value for key, value in launch["source_sha256"].items()},
                  "windows": {}, "memory_per_rank": None, "policies": []}
        for window in (1, 2):
            request_path = raw / label / f"w{window}_requests.jsonl"
            if not request_path.exists():
                continue
            rows = [json.loads(line) for line in request_path.read_text().splitlines()]
            agg = json.loads((raw / label / f"w{window}_aggregate.json").read_text())
            import collections
            record["windows"][f"w{window}"] = {
                "requests": len(rows), "output_tokens": sum(r["output_tokens"] for r in rows),
                "median_TPOT_ms": statistics.median(r["tpot_ms"] for r in rows),
                "median_TTFT_ms": statistics.median(r["ttft_ms"] for r in rows),
                "p95_latency_ms": p95([r["request_latency_ms"] for r in rows]),
                "output_tokens_per_s": agg["output_tokens_per_s"],
                "first_token_timestamp_groups": sorted(collections.Counter(r["first_token_ts_s"] for r in rows).values()),
                "per_request": [{key: r[key] for key in ("request_idx", "prompt_tokens", "output_tokens",
                    "ttft_ms", "tpot_ms", "request_latency_ms")} for r in rows]}
        for policy_path in sorted((raw / label).glob("rank*_policy.local.json")):
            policy = json.loads(policy_path.read_text())
            record["policies"].append({key: policy[key] for key in
                ("rank", "mode", "converted", "conversion_wall_s", "weight_nz_mode", "allocated_bytes_after_load", "peak_allocated_bytes_after_load")})
            record["policies"][-1]["formats_after"] = sorted({w["after"] for w in policy["qkv_weights"]})
        memory_path = raw / label / "memory.local.json"
        if memory_path.exists():
            record["memory_per_rank"] = json.loads(memory_path.read_text())
        record["performance_admission"] = "NOT_ADMISSIBLE_NO_EQUIVALENT_COMPLETED_TRIPLET"
        processes.append(record)
    return {"formal_approved": False, "status": "M0_INCONCLUSIVE_FULL_OUTPUT_REPRODUCIBILITY_BLOCKED",
            "completed_matched_triplets": 0, "accepted_optimization_gains": [],
            "profiler": "NOT_RUN_FAIL_STOP", "processes": processes}


def analyze(raw, labels):
    pairs = [read_arm(raw, label) for label in labels]
    assert pairs[0][1] == pairs[1][1] == pairs[2][1], "executed source/input identity drift"
    arms = [pair[0] for pair in pairs]
    assert [a["arm"] for a in arms] == ["B0", "B1", "B0"]
    assert outputs(raw, labels[0]) == outputs(raw, labels[1]) == outputs(raw, labels[2])
    drift = {}
    for window in ("w1", "w2"):
        pre, candidate, post = [a["windows"][window] for a in arms]
        drift[window] = abs(post["median_TPOT_ms"] / pre["median_TPOT_ms"] - 1) * 100
    pre, candidate, post = [a["windows"]["w2"] for a in arms]
    reference = (pre["median_TPOT_ms"] + post["median_TPOT_ms"]) / 2
    gain = (1 - candidate["median_TPOT_ms"] / reference) * 100
    valid = drift["w2"] < 2
    return {"status": "VALID_LOCAL_ONLY" if valid else "INVALID_ROLLBACK_DRIFT",
            "formal_approved": False, "full_token_and_text_oracle": "PASS_32_REQUESTS_ALL_ARMS",
            "rollback_drift_percent": drift, "primary": "UNPROFILED_W2_MEDIAN_REQUEST_TPOT",
            "raw_service_TPOT_gain_percent": gain,
            "accepted_service_gain_percent": gain if valid else None,
            "throughput_change_percent": 100 * (candidate["output_tokens_per_s"] /
                ((pre["output_tokens_per_s"] + post["output_tokens_per_s"]) / 2) - 1),
            "note": "TP4 service, not synthetic component-chain gain; profiler excluded; descriptive local reproduction",
            "source_and_input_fingerprint": hashlib.sha256(
                json.dumps(list(pairs[0][1].values())).encode()).hexdigest(),
            "arms": arms}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--campaign-summary", action="store_true")
    args = parser.parse_args()
    if args.campaign_summary:
        result = campaign_summary(args.raw)
    else:
        assert len(args.labels) == (2 if args.oracle_only else 3)
        result = token_report(args.raw, args.labels) if args.oracle_only else analyze(args.raw, args.labels)
    assert not args.out.exists(), "do not overwrite evidence"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("arms", "processes", "differences")}, indent=2))
