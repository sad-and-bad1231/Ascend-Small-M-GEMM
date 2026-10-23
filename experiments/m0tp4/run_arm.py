#!/usr/bin/env python3
"""Minimal PR14 prospective adapter: original workload/metrics, full token custody."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    assert os.environ.get("M0_OWNER_TP4_AUTHORIZED") == "1"
    assert os.environ.get("VLLM_ASCEND_QKV_NZ_MODE") == ("1" if args.arm == "B1" else "0")
    sys.path.insert(0, str(Path(args.pr14) / "scripts"))
    import qkv_same_run_capture as historical
    import torch
    import vllm
    import vllm_ascend
    from vllm import LLM, SamplingParams
    from qkv_policy import memory_snapshot

    assert torch.npu.device_count() == 4
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["M0_ARM_OUT"] = str(out)
    config = dict(model=args.model, tensor_parallel_size=4, dtype="bfloat16",
                  max_num_seqs=8, max_model_len=4096, gpu_memory_utilization=0.9,
                  disable_log_stats=False, seed=0,
                  compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY",
                                      "cudagraph_capture_sizes": [1, 2, 4, 8],
                                      "max_cudagraph_capture_size": 8})
    if args.profile:
        config["profiler_config"] = {
            "profiler": "torch", "torch_profiler_dir": str(out / "trace"),
            "ignore_frontend": True, "torch_profiler_with_stack": False,
            "delay_iterations": 4, "max_iterations": 5}
    identity = {"arm": args.arm, "profile": args.profile, "formal_approved": False,
                "reproduction": "CLEAN_PROSPECTIVE_CURRENT_RUNTIME",
                "runner_sha256": sha(__file__), "policy_sha256": sha(HERE / "qkv_policy.py"),
                "historical_runner_sha256": sha(Path(args.pr14) / "scripts/qkv_same_run_capture.py"),
                "vllm_path": vllm.__file__, "plugin_path": vllm_ascend.__file__,
                "trace_sha256": sha(args.prompts), "config": config}
    (out / "identity.local.json").write_text(json.dumps(identity, indent=2))
    prompts = historical._load_prompts(Path(args.prompts))
    start = time.perf_counter()
    llm = LLM(**config)
    load_s = time.perf_counter() - start
    sampling = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True, seed=0)
    (out / "resolved_config.local.txt").write_text(str(llm.llm_engine.vllm_config))
    summaries, all_records = [], []
    for window, name, offset, count in ((1, "w1_capture", 0, 8), (2, "w2_steady", 8, 24)):
        batch = prompts[offset:offset + count]
        profiled = args.profile and window == 2
        if profiled:
            llm.start_profile(profile_prefix=f"{args.arm}_{name}")
        start = time.perf_counter()
        outputs = llm.generate(batch, sampling, use_tqdm=False)
        wall_s = time.perf_counter() - start
        if profiled:
            llm.stop_profile()
        assert len(outputs) == count
        rows, records = [], []
        for idx, output in enumerate(outputs):
            assert output.finished and len(output.outputs) == 1
            assert len(output.outputs[0].token_ids) == 64
            assert output.prompt == batch[idx], "returned request order differs"
            row = historical._row_from_output(args.arm, window, name, idx, output)
            assert row["last_token_ts_s"] > row["first_token_ts_s"]
            # Arrival is wall-clock, token timestamps are monotonic: sum intervals.
            row["request_latency_ms"] = 1000 * (output.metrics.first_token_latency +
                output.metrics.last_token_ts - output.metrics.first_token_ts)
            assert row["request_latency_ms"] > 0
            rows.append(row)
            record = {"window": window, "request_idx": idx, "trace_idx": offset + idx,
                      "prompt": batch[idx], "prompt_token_ids": list(output.prompt_token_ids),
                      "finished": output.finished,
                      "output_token_ids": list(output.outputs[0].token_ids),
                      "output_text": output.outputs[0].text}
            record["payload_sha256"] = hashlib.sha256(
                json.dumps(record, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
            records.append(record)
        assert not historical._validate_rows(rows)
        aggregate = historical._aggregate(rows, args.arm, window, name,
                                           "PROFILED" if profiled else "UNPROFILED", wall_s)
        aggregate["definition"] = "PR14 engine request stats; profiling status explicit; not HTTP latency"
        aggregate["output_tokens_total"] = sum(r["output_tokens"] for r in rows)
        aggregate["output_tokens_per_s"] = aggregate["output_tokens_total"] / wall_s
        aggregate["request_latency_ms"] = [r["request_latency_ms"] for r in rows]
        aggregate["p95_definition"] = "nearest-rank ceil(0.95*n); TTFT+engine decode span, not HTTP end-to-end"
        for suffix, values in (("requests", rows), ("outputs.local", records)):
            (out / f"w{window}_{suffix}.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in values))
        (out / f"w{window}_aggregate.json").write_text(json.dumps(aggregate, indent=2))
        all_records.extend(records)
        summaries.append(aggregate)
        print(f"M0_WINDOW {args.arm} w{window} profiled={profiled} tpot={aggregate['tpot_ms']} wall={wall_s}", flush=True)
    policies = [json.loads((out / f"rank{i}_policy.local.json").read_text()) for i in range(4)]
    assert all(p["converted"] == (64 if args.arm == "B1" else 0) for p in policies)
    if args.reference:
        reference = []
        for window in (1, 2):
            reference.extend(json.loads(line) for line in
                             (args.reference / f"w{window}_outputs.local.jsonl").read_text().splitlines())
        differences = [i for i, (a, b) in enumerate(zip(reference, all_records)) if a != b]
        oracle = {"records": len(all_records), "reference_records": len(reference),
                  "differing_indices": differences, "exact_match": reference == all_records}
        (out / "oracle.json").write_text(json.dumps(oracle, indent=2))
        assert oracle["exact_match"], f"full output oracle failed: {differences}"
    memory = llm.collective_rpc(memory_snapshot)
    (out / "memory.local.json").write_text(json.dumps(memory, indent=2))
    assert all(m["peak_allocated_bytes"] <= 58 * 2**30 for m in memory), "HBM budget exceeded"
    inventory = {"arm": args.arm, "profile": args.profile, "load_and_graph_init_s": load_s,
                 "windows": summaries, "memory": memory, "records": len(all_records),
                 "raw_output_sha256": {f"w{i}": sha(out / f"w{i}_outputs.local.jsonl") for i in (1, 2)}}
    (out / "inventory.local.json").write_text(json.dumps(inventory, indent=2))
    print("M0_ARM_COMPLETE", flush=True)
    # PR14 process lifecycle: parent owns session cleanup and checks HBM release.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["B0", "B1"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--pr14", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--profile", action="store_true")
    run(parser.parse_args())
