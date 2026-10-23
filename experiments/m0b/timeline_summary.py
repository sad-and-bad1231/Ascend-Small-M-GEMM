"""Device-timestamp timeline reducer; no summation of overlapping streams."""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import statistics

from trace_summary import trace_events


def timeline(tasks, kernels, expected_weight_format="ND"):
    if expected_weight_format not in ("ND", "FRACTAL_NZ"):
        raise ValueError("unsupported expected weight format")
    ordered = sorted(tasks, key=lambda r: Decimal(r["task_start(us)"].strip()))
    origin = Decimal(ordered[0]["task_start(us)"].strip())
    kernel_index = {(Decimal(r["Start Time(us)"].strip()), r["Stream ID"], r["Task ID"]): r
                    for r in kernels}
    public = []
    for row in ordered:
        start, stop = (Decimal(row[k].strip()) for k in ("task_start(us)", "task_stop(us)"))
        if stop < start or abs(stop - start - Decimal(row["task_time(us)"].strip())) > Decimal("0.002"):
            raise ValueError("inconsistent task timestamp/duration")
        kernel = kernel_index.get((start, row["stream_id"], row["task_id"]))
        public.append({"sequence_by_start": len(public), "kernel": row["kernel_name"],
                       "type": row["kernel_type"], "stream_id": row["stream_id"],
                       "task_id": row["task_id"], "model_id": kernel["Model ID"] if kernel else None,
                       "dispatch": kernel["Name"] if kernel else None,
                       "block_num": kernel["Block Num"] if kernel else None,
                       "input_formats": kernel["Input Formats"] if kernel else None,
                       "input_dtypes": kernel["Input Data Types"] if kernel else None,
                       "start_us_relative": float(start - origin),
                       "stop_us_relative": float(stop - origin), "duration_us": float(stop - start)})
    models = [r for r in public if r["type"] == "MODEL_EXECUTE"]
    if len(models) != 5 or len(kernels) != 10:
        raise ValueError("need exactly five model executions and ten compute kernels")
    replays = []
    for i, model in enumerate(models):
        end = models[i + 1]["start_us_relative"] if i < 4 else float("inf")
        group = [r for r in public if model["start_us_relative"] <= r["start_us_relative"] < end
                 and not r["type"].startswith("PROFILING_")]
        gemms = [r for r in group if r["type"] == "AI_CORE"]
        consumers = [r for r in group if r["type"] == "AI_VECTOR_CORE"]
        if len(gemms) != 1 or len(consumers) != 1:
            raise ValueError("ambiguous replay compute association")
        gemm, consumer = gemms[0], consumers[0]
        if (gemm["input_formats"] != "ND;" + expected_weight_format or gemm["input_dtypes"] != "DT_BF16;DT_BF16"
                or consumer["kernel"] != "split_qkv_rmsnorm_rope_kernel"
                or consumer["start_us_relative"] < gemm["stop_us_relative"]):
            raise ValueError("unexpected compute path/layout/order")
        if any(any(token in (r["kernel"] + r["type"]).lower()
                   for token in ("transdata", "formatcast", "memcpy", "transpose")) for r in group):
            raise ValueError("unexpected conversion/copy task")
        finish = max(r["stop_us_relative"] for r in group)
        replays.append({"replay": i + 1,
                        "gemm_us": gemm["duration_us"], "consumer_us": consumer["duration_us"],
                        "handoff_gap_us": consumer["start_us_relative"] - gemm["stop_us_relative"],
                        "graph_envelope_us": finish - model["start_us_relative"],
                        "prefix_to_gemm_us": gemm["start_us_relative"] - model["start_us_relative"],
                        "completion_tail_us": finish - consumer["stop_us_relative"],
                        "model_execute_task_us": model["duration_us"],
                        "notify_wait_task_us": sum(r["duration_us"] for r in group if r["type"] == "NOTIFY_WAIT"),
                        "notify_record_task_us": sum(r["duration_us"] for r in group if r["type"] == "NOTIFY_RECORD"),
                        "task_sequences": [r["sequence_by_start"] for r in group]})
    return {"tasks_in_start_order": public, "replays": replays,
            "medians_us": {k: statistics.median(r[k] for r in replays) for k in replays[0] if k.endswith("_us")},
            "task_counts": dict(Counter(r["type"] for r in public)),
            "association": "DEVICE_ORDER_AND_KERNEL_TIMESTAMP_JOIN_NOT_ACL_FLOW_PROOF",
            "graph_envelope_definition": "MODEL_EXECUTE start to last non-profiling device task stop before next model",
            "cross_stream_task_durations_not_additive": True}


def summarize(run):
    run = Path(run)
    record = json.loads((run / "result.local.json").read_text())
    if ((run.parent / (run.name + ".exit.local.txt")).read_text().strip() != "0"
            or record["stage"] != "timeline" or record["arm"] != "B0" or record["weight_format"] != 2
            or record["mode"] != "DIAGNOSTIC_ONLY" or record["formal_approval"] is not False
            or record["shared"]["m"] != 1 or record["shared"]["seed"] != 0 or record["shared"]["bias"]):
        raise ValueError("failed or outside authorized timeline scope")
    files = {}
    for name in ("task_time.csv", "kernel_details.csv", "trace_view.json"):
        matches = list((run / "profiler").rglob("ASCEND_PROFILER_OUTPUT/" + name))
        if len(matches) != 1:
            raise ValueError("missing/ambiguous profiler export")
        files[name] = matches[0]
    data = timeline(list(csv.DictReader(files["task_time.csv"].open())),
                    list(csv.DictReader(files["kernel_details.csv"].open())))
    phases = record["timeline_windows"]
    if [r["phase"] for r in phases] != ["before", "after"] or "post_profile" not in record["correctness"]:
        raise ValueError("missing bracketing timing or full profile oracle")
    for phase in phases:
        obs = phase["observation"]
        if (len(phase["trials"]) != 5 or not phase["oracle"] or not obs["weight_pointer_unchanged"]
                or obs["weight_format_after"] != 2 or obs["peak_allocated"] > 2 * 1024**3
                or [obs[k] for k in ("warmup_replays", "trials", "replays_per_trial")] != [10, 5, 100]):
            raise ValueError("timing/oracle/budget protocol violation")
        phase["medians_us"] = {k: statistics.median(t[k] for t in phase["trials"]) for k in ("event_us", "wall_us")}
        del phase["oracle"]
        phase["oracle_status"] = "PASS_FULL_AND_EXACT"
    drift = {k: (phases[1]["medians_us"][k] - phases[0]["medians_us"][k]) /
             phases[0]["medians_us"][k] * 100 for k in ("event_us", "wall_us")}
    events = trace_events(json.loads(files["trace_view.json"].read_text()))
    markers = Counter(e.get("name") for e in events if e.get("ph") == "X")
    data.update({"run": run.name, "formal_approved": False, "phases": phases,
                 "before_after_signed_drift_percent": drift,
                 "profiler_interference": "FLAG_GE_2PCT" if any(abs(v) >= 2 for v in drift.values()) else "NOT_DETECTED_BY_2PCT_GATE_NOT_PROOF_OF_NO_EFFECT",
                 "executed_probe_sha256": record["shared"]["probe_sha"],
                 "full_profile_oracle": "PASS", "B0_pair_errors": record["correctness"].get("B0_pair"),
                 "graph_markers": {k: markers[k] for k in ("AscendCL@aclmdlRIExecuteAsync", "MODEL_EXECUTE")},
                 "raw_sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in
                                 {**files, "result.local.json": run / "result.local.json"}.items()}})
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(summarize(parser.parse_args().run), indent=2))
