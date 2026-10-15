#!/usr/bin/env python3
"""Summarize exactly one preregistered B0/B1/B0 pilot; never filter trials."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROLES = ("B0_before", "B1", "B0_after")


def assess(trials):
    metrics = {}
    for metric in ("event_us", "wall_us"):
        values = [[trial[metric] for trial in arm] for arm in trials]
        if any(len(arm) != 5 or not all(math.isfinite(v) and v > 0 for v in arm) for arm in values):
            raise ValueError("need all five finite positive trials for each arm")
        before, candidate, after = [statistics.median(arm) for arm in values]
        reference = (before + after) / 2
        drift = (after - before) / before * 100
        metrics[metric] = {
            "medians": dict(zip(ROLES, (before, candidate, after))),
            "trial_ranges": {role: [min(arm), max(arm)] for role, arm in zip(ROLES, values)},
            "B0_reference": reference,
            "rollback_signed_drift_percent": drift,
            "rollback_absolute_drift_percent": abs(drift),
            "raw_latency_reduction_percent": (reference - candidate) / reference * 100,
        }
    invalid = any(m["rollback_absolute_drift_percent"] >= 2 for m in metrics.values())
    for metric in metrics.values():
        metric["drift_screened_point_estimate_percent"] = (
            None if invalid else metric["raw_latency_reduction_percent"])
    return {"status": "INVALID_DRIFT" if invalid else "PILOT_DRIFT_SCREEN_PASS",
            "gate": "Event OR Wall absolute median rollback drift >=2% invalidates whole triplet",
            "metrics": metrics, "formal_approved": False,
            "independent_allocations_per_arm": 1, "statistical_significance": "NOT_ESTABLISHED"}


def summarize(root, m):
    paths = [Path(root) / f"m{m}_{role}" for role in ROLES]
    records = [json.loads((path / "result.local.json").read_text()) for path in paths]
    if not all(record["shared"] == records[0]["shared"] for record in records):
        raise ValueError("input/source/runtime/probe identity mismatch")
    for path, record, arm in zip(paths, records, ("B0", "B1", "B0")):
        if (path.parent / (path.name + ".exit.local.txt")).read_text().strip() != "0":
            raise ValueError("nonzero or missing process exit receipt")
        if (record["stage"] != "timing" or record["scope"] != "chain"
                or record["mode"] != "DIAGNOSTIC_ONLY" or record["formal_approval"] is not False
                or record["arm"] != arm or record["shared"]["m"] != m
                or record["shared"]["seed"] != 0 or record["shared"]["bias"]):
            raise ValueError("outside authorized pilot")
        if record["profile_observation"] is not None or (path / "profiler").exists():
            raise ValueError("profiler evidence in a timing process")
        observation = record["timing_observation"]
        if ([observation[k] for k in ("warmup_replays", "trials", "replays_per_trial")] != [10, 5, 100]
                or not observation["weight_pointer_unchanged"] or observation["peak_allocated"] > 2 * 1024**3
                or "post_timing" not in record["correctness"]):
            raise ValueError("timing protocol/oracle/resource failure")
    summary = assess([r["trials"] for r in records])
    summary.update({"m": m, "seed": 0, "mode": "DIAGNOSTIC_ONLY", "scope": "chain",
                    "executed_probe_sha256": records[0]["shared"]["probe_sha"],
                    "matched_input_source_runtime_fingerprints": True,
                    "oracle": "PASS_BEFORE_AND_AFTER_TIMING", "profiler": "NOT_RUN",
                    "arms": [{"role": role, "receipt": path.name, "exit_code": 0,
                              "trials": record["trials"], "observation": record["timing_observation"],
                              "B0_pair_errors": record["correctness"].get("B0_pair"),
                              "raw_result_sha256": hashlib.sha256((path / "result.local.json").read_bytes()).hexdigest()}
                             for role, path, record in zip(ROLES, paths, records)]})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--m", type=int, choices=(1, 8), required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.m), indent=2))
