#!/usr/bin/env python3
"""CPU-only M0 headroom accounting from hash-bound existing device traces."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from timeline_summary import timeline


def headroom(m, span_us, gap_us):
    if m not in (1, 8) or span_us <= 0 or gap_us < 0 or gap_us > span_us:
        raise ValueError("outside the retained diagnostic cells")
    return {
        "m": m,
        "intermediate_qkv_bytes": 2*m*2560,
        "optimistic_avoidable_write_plus_read_bytes": 4*m*2560,
        "mandatory_final_q_k_v_output_bytes": 2*m*2560,
        "logical_weight_bytes_not_removed": 2*5120*2560,
        "scenario_five_percent_of_profiled_median_span_us": .05*span_us,
        "scenario_gap_only_span_reduction_percent": 100*gap_us/span_us,
        "scenario_additional_savings_needed_beyond_gap_us": max(0, .05*span_us-gap_us),
        "mandatory_consumer_math_is_not_free": True,
        "byte_count_is_not_measured_GM_HBM_traffic_or_latency": True,
        "profiled_median_scenario_is_not_formal_p95_or_service_gain": True,
    }


def retained_profile(root, m, arm):
    label = f"profile_m{m}_{arm}"
    run = root / "m0_receipts/m0b_dispatch_20261015/resumed" / label
    published = root / "results/m0b_dispatch_20261015" / (label + ".json")
    receipt = json.loads(published.read_text())
    result_path = run / "result.local.json"
    result = json.loads(result_path.read_text())
    assert result["stage"] == "profile" and result["mode"] == "DIAGNOSTIC_ONLY"
    assert result["formal_approval"] is False and result["arm"] == arm
    assert result["shared"]["m"] == m and result["shared"]["seed"] == 0
    assert not result["shared"]["bias"]
    assert (run.parent / (label + ".exit.local.txt")).read_text().strip() == "0"
    obs = result["profile_observation"]
    assert obs["replays"] == 5 and obs["weight_pointer_unchanged"]
    assert obs["peak_allocated"] < 2*1024**3
    files = {"result.local.json": result_path}
    for name in ("task_time.csv", "kernel_details.csv", "trace_view.json"):
        matches = list((run / "profiler").rglob("ASCEND_PROFILER_OUTPUT/" + name))
        assert len(matches) == 1
        files[name] = matches[0]
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    assert hashes == receipt["raw_sha256"], "raw custody mismatch; stop reduction"
    fmt = "ND" if arm == "B0" else "FRACTAL_NZ"
    with files["task_time.csv"].open() as tasks, files["kernel_details.csv"].open() as kernels:
        data = timeline(list(csv.DictReader(tasks)), list(csv.DictReader(kernels)), fmt)
    expected_dispatch = "aclnnMatmul_MatMulCommon_MatMulV2" if arm == "B0" else "aclnnMatmulWeightNz_MatMulCommon_MatMulV2"
    assert all(t["dispatch"] == expected_dispatch for t in data["tasks_in_start_order"] if t["type"] == "AI_CORE")
    medians = data["medians_us"]
    data.update(arm=arm, m=m, seed=0, formal_approved=False,
                evidence_scope="SYNTHETIC_PER_RANK_GRAPH_NOT_REAL_TP4_SERVICE",
                historical_raw_sha256_verified=hashes,
                published_receipt_sha256=hashlib.sha256(published.read_bytes()).hexdigest(),
                accounting=headroom(m, medians["graph_envelope_us"], medians["handoff_gap_us"]),
                cross_process_arm_speedup="NOT_ESTIMATED_NO_MATCHED_STABILITY_CONTROL",
                correctness="RETAINED_FROZEN_COMPONENT_ORACLE_NOT_MODEL_TOKEN_OR_FORMAL_COSINE_CERTIFICATION")
    return data


def build(root):
    profiles = [retained_profile(root, m, arm) for m in (1, 8) for arm in ("B0", "B1")]
    timeline_panel = []
    for index in (1, 2, 3):
        path = root / f"results/m0b_timeline_20261017/process{index}.json"
        row = json.loads(path.read_text())
        medians = row["medians_us"]
        timeline_panel.append({"process": index, "medians_us": medians,
            "source_receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "accounting": headroom(1, medians["graph_envelope_us"], medians["handoff_gap_us"])})
    return {"base": "e8c7bbb74e1643438f94fd609bfb8b690a4d367d", "formal_approved": False,
            "new_hardware_processes": 0, "new_device_replays": 0,
            "decision": "M0_INCONCLUSIVE_REVIEW_CLOSURE_NOT_MECHANISM_FALSIFICATION",
            "historical_penalty_only_scenario": {
                "approximate_consumer_increase_percent": 2.3,
                "max_candidate_chain_reduction_percent_if_only_that_penalty_removed": 100*2.3/102.3,
                "assumptions": "SAME_TIME_STATISTIC_UNCHANGED_GEMM_AND_OTHER_COSTS_CONSUMER_SHARE_LE_1",
                "not_a_formal_p95_bound_or_measured_B2_result": True,
                "historical_CSV_revalidation": "NOT_RUN_RETAINED_HANDOFF_VALUE_ONLY"},
            "profiles": profiles, "existing_B0_timeline_panel": timeline_panel,
            "service_scenario": {"model": "SERIAL_LATENCY_REDUCTION_APPROXIMATION_NOT_THROUGHPUT_PREDICTION",
                "fraction_required_if_component_latency_reduction_5pct": .02/.05,
                "fraction_required_if_component_latency_reduction_10pct": .02/.10,
                "fraction_required_if_component_latency_reduction_20pct": .02/.20,
                "actual_service_critical_path_fraction": "UNKNOWN",
                "measured_fusion_savings": "UNKNOWN_NO_B2_OR_ISOLATED_OVERHEAD_CONTROL"}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(args.root), indent=2) + "\n")
