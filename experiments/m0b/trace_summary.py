#!/usr/bin/env python3
"""Sanitize this M0-B trace's labels/counts/formats; deliberately omit timings/IDs."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def trace_events(raw):
    return raw if isinstance(raw, list) else raw["traceEvents"]


def summarize(run):
    run = Path(run)
    result_path = run / "result.local.json"
    result = json.loads(result_path.read_text())
    files = {}
    for name in ("kernel_details.csv", "task_time.csv", "trace_view.json"):
        matches = list((run / "profiler").rglob("ASCEND_PROFILER_OUTPUT/" + name))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {name}, got {len(matches)}")
        files[name] = matches[0]
    kernels = list(csv.DictReader(files["kernel_details.csv"].open()))
    tasks = list(csv.DictReader(files["task_time.csv"].open()))
    events = trace_events(json.loads(files["trace_view.json"].read_text()))
    if not kernels or not tasks:
        raise ValueError("empty kernel or task evidence")
    keys = ("Name", "Type", "Input Shapes", "Input Data Types", "Input Formats",
            "Output Formats", "Accelerator Core", "Block Num")
    groups = Counter(tuple(row[key] for key in keys) for row in kernels)
    task_groups = Counter((row["kernel_name"], row["kernel_type"]) for row in tasks)
    conversion_tokens = ("transdata", "formatcast", "memcpy", "transpose", "cast")
    conversion_tasks = [{"kernel": name, "type": kind, "count": count}
                        for (name, kind), count in task_groups.items()
                        if any(token in (name + kind).lower() for token in conversion_tokens)]
    names = Counter(e.get("name", "") for e in events if e.get("ph") == "X")
    console = run.parent / (run.name + ".console.local.txt")
    text = console.read_text() if console.exists() else ""
    return {
        "receipt": run.name, "arm": result["arm"], "m": result["shared"]["m"],
        "mode": result["mode"], "formal_approved": result["formal_approval"],
        "executed_probe_sha256": result["shared"]["probe_sha"],
        "weight_format": result["weight_format"], "profile": result["profile_observation"],
        "kernels": [{**dict(zip(keys, key)), "count": count} for key, count in groups.items()],
        "native_tasks": [{"kernel": name, "type": kind, "count": count}
                         for (name, kind), count in task_groups.items()],
        "compute_launches": sum(count for (_, kind), count in task_groups.items()
                                if kind in ("AI_CORE", "AI_VECTOR_CORE")),
        "graph_events": {name: count for name, count in names.items()
                         if name.startswith("M0B_GRAPH_REPLAY_") or name in
                         ("AscendCL@aclmdlRIExecuteAsync", "MODEL_EXECUTE")},
        "standalone_conversion_candidates": conversion_tasks,
        "copy_or_conversion_duration_events": {
            name: count for name, count in names.items()
            if any(token in name.lower() for token in conversion_tokens)},
        "host_to_device_flow_markers_NOT_memcpy": sum(
            e.get("cat") == "HostToDevice" and e.get("ph") in ("s", "f") for e in events),
        "parser_limits": {
            "acl_to_npu_flow_missing": "Failed to get acl to npu flow events" in text,
            "op_attr_ignored_with_text_export": "op_attr switch is invalid" in text,
            "schedule_warning": "Incorrect schedule:" in text},
        "gemm_fp32_rel_l2": result["correctness"]["post_profile"]["gemm_fp32"]["rel_l2"],
        "B0_pair_errors": result["correctness"].get("B0_pair"),
        "raw_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for name, path in {"result.local.json": result_path, **files}.items()},
        "kernel_internal_layout_work": "UNKNOWN_NOT_A_DISASSEMBLY_OR_COUNTER_STUDY",
        "timing_benefit": "NOT_MEASURED",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run), indent=2))
