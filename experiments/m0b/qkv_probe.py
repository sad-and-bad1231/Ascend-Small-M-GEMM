#!/usr/bin/env python3
"""Isolated M0-B preparation. Default path is stdlib-only and never touches NPU.

Reuses PR14's f_linear_nz policy and graph/event measurement method, not its
TP4 runner or environment activation. Formal admission is separate from bounded
user-authorized diagnostics; neither mode disables safety or correctness checks.
"""

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CONSUMER_MODULE = "vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_rope"
CONSUMER_SHA = "ca226aac8c5559904646765e3f358dcdd394c8d8af58f5102338676844312ae1"
SOURCE_SHAS = {
    "/vllm-workspace/vllm": "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665",
    "/vllm-workspace/vllm-ascend": "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd",
}
M_VALUES = (1, 2, 4, 8, 16)
ATOL, RTOL = 0.05, 0.005


def weight_format(arm, m, bias=False, shape=(2560, 5120)):
    if arm not in ("B0", "B1") or m not in M_VALUES or shape != (2560, 5120):
        raise ValueError("outside frozen QKV matrix")
    return 29 if arm == "B1" and m in (1, 2, 4, 8) and not bias else 2


def plan():
    return {
        "status": "PREPARATION_ONLY_HARDWARE_NOT_RUN",
        "shape": {"M": list(M_VALUES), "K": 5120, "N": 2560, "dtype": "BF16"},
        "heads": {"q": 16, "kv": 2, "dim": 128},
        "cells": [{"m": m, "bias": bias, "B0_format": 2,
                   "B1_format": weight_format("B1", m, bias)}
                  for m, bias in [(1, False), (2, False), (4, False), (8, False),
                                  (16, False), (1, True)]],
        "order": "fresh processes B0/B1/B0, seed=0..9",
        "timing": {"warmups": 10, "trials": 5, "replays_per_trial": 100},
        "hardware_gate": "FORMAL requires Issue20 review; DIAGNOSTIC_ONLY permits the user-authorized bounded correctness cell",
    }


def require_approval(manifest):
    approval = manifest["approval"]
    link = approval.get("advisor_review_link", "")
    if approval.get("approved_for_hardware") is not True or not link.startswith(
        "https://github.com/intellistream/ascend-small-m-gemm/issues/20#issuecomment-"
    ):
        raise RuntimeError("BLOCKED: formal experiments require reviewed Issue20 approval before runtime import")


def require_execution_permission(manifest, args):
    if getattr(args, "stage", None) == "stability":
        permission = manifest.get("stability_permission", {})
        if (permission.get("user_authorized") is not True
                or getattr(args, "mode", None) != "DIAGNOSTIC_ONLY"
                or args.arm != "B0" or args.m != 1 or args.seed != 0 or args.bias
                or args.scope != "chain" or getattr(args, "windows", 1) not in (1, 3)
                or getattr(args, "physical_card", None) is None
                or not getattr(args, "device_identity_sha", None)):
            raise RuntimeError("BLOCKED: outside separately authorized B0 stability scope")
        return
    if getattr(args, "mode", "FORMAL") == "DIAGNOSTIC_ONLY":
        permission = manifest.get("diagnostic_permission", {})
        if permission.get("user_authorized") is not True:
            raise RuntimeError("BLOCKED: allocated-device diagnostic authorization required")
        allowed = any(args.m == cell["m"] and args.bias == cell["bias"]
                      and args.stage in cell["stages"] for cell in permission.get("cells", []))
        if args.scope != "chain" or args.seed != 0 or not allowed:
            raise RuntimeError("BLOCKED: outside the authorized diagnostic cell/stage/seed")
        return
    require_approval(manifest)


def consumer_reference(qkv, cache, positions, qw, kw, head_dim=128, q_heads=16,
                       kv_heads=2, eps=1e-6):
    """CPU FP32 split/headwise RMSNorm/NeoX RoPE, like the pinned op test.

    No intermediate BF16 norm rounding: this deliberately remains an independent
    high-precision reference, not a transliteration of the Triton kernel.
    """
    import torch
    if any(t.device.type != "cpu" for t in (qkv, cache, positions, qw, kw)):
        raise ValueError("oracle is CPU-only")
    m = qkv.shape[0]
    q, k, v = qkv.float().split([q_heads * head_dim, kv_heads * head_dim,
                                kv_heads * head_dim], -1)
    cos, sin = cache.float()[positions].chunk(2, -1)
    cos, sin = cos[:, None, :], sin[:, None, :]

    def norm_rope(x, w):
        x = x.reshape(m, -1, head_dim)
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()
        a, b = x.chunk(2, -1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1).reshape(m, -1)

    return norm_rope(q, qw), norm_rope(k, kw), v


def compare(actual, expected, atol=ATOL, rtol=RTOL, rel_l2_limit=None):
    import torch
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("shape mismatch or nonfinite oracle input")
    d = a - b
    metrics = {"max_abs": d.abs().max().item(),
               "max_rel": (d.abs() / b.abs().clamp_min(1e-8)).max().item(),
               "rel_l2": (d.norm() / b.norm().clamp_min(1e-8)).item()}
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    if rel_l2_limit is not None and metrics["rel_l2"] > rel_l2_limit:
        raise AssertionError(f"relative L2 gate: {metrics}")
    return metrics


def digest_tensor(tensor):
    import torch
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def initialize_isolated_runtime(npu, triton_utils):
    # Match worker/model_runner startup without importing the TP service runner.
    # This permits internal storage; it does NOT select global weight_nz_mode=2.
    npu.config.allow_internal_format = True
    triton_utils.init_device_properties_triton()


def hardware(args, manifest, out):
    # Execution permission is not formal research approval. Both precede runtime.
    require_execution_permission(manifest, args)

    def checkpoint(stage, **fields):
        entry = {"stage": stage, **fields}
        with (out / "progress.local.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    checkpoint("permission", mode=args.mode, formal_approval=manifest["approval"]["approved_for_hardware"])
    if args.stage == "stability":
        from stability_state import snapshot as state_snapshot, validate as validate_state
        preflight = state_snapshot(args.physical_card)
        (out / "pre_runtime_state.local.json").write_text(json.dumps(preflight, indent=2))
        validate_state(preflight, args.device_identity_sha)
        checkpoint("physical_identity_and_occupancy_verified_before_runtime_import")
    import torch
    import torch_npu
    import torch.nn.functional as F
    from importlib.metadata import version
    if os.environ.get("VLLM_ASCEND_ENABLE_NZ") == "2":
        raise RuntimeError("global NZ mode is not B1")
    for root, expected in SOURCE_SHAS.items():
        if subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip() != expected:
            raise RuntimeError("source identity changed; re-review contract")
        if subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip():
            raise RuntimeError("golden source is dirty")
    if torch.npu.device_count() != 1:
        raise RuntimeError("exactly one allocated visible NPU required")
    torch.npu.set_device(0)
    module = importlib.import_module(CONSUMER_MODULE)
    source = Path(module.__file__).resolve()
    if source != Path("/vllm-workspace/vllm-ascend", *CONSUMER_MODULE.split(".")).with_suffix(".py"):
        raise RuntimeError("consumer imported from unexpected path")
    if hashlib.sha256(source.read_bytes()).hexdigest() != CONSUMER_SHA:
        raise RuntimeError("consumer changed")
    consumer = module.split_qkv_rmsnorm_rope_impl
    initialize_isolated_runtime(torch.npu, importlib.import_module("vllm_ascend.ops.triton.triton_utils"))
    checkpoint("source_and_consumer_verified", consumer_sha=CONSUMER_SHA)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x_cpu = torch.randn(args.m, 5120, generator=gen, device="cpu").bfloat16()
    w_cpu = (torch.randn(2560, 5120, generator=gen, device="cpu") / math.sqrt(5120)).bfloat16()
    qw_cpu = torch.randn(128, generator=gen, device="cpu").bfloat16()
    kw_cpu = torch.randn(128, generator=gen, device="cpu").bfloat16()
    bias_cpu = torch.randn(2560, generator=gen, device="cpu").bfloat16() if args.bias else None
    positions_cpu = torch.arange(args.m, device="cpu", dtype=torch.int64) * 257
    inv_freq = 1e6 ** (-torch.arange(0, 128, 2, device="cpu").float() / 128)
    angles = torch.arange(4096, device="cpu").float()[:, None] * inv_freq
    cache_cpu = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    shared = {"seed": args.seed, "m": args.m, "bias": args.bias,
              "sources": SOURCE_SHAS, "consumer_sha": CONSUMER_SHA,
              "process_config": {"allow_internal_format": True},
              "runtime": {name: version(name) for name in
                          ("torch", "torch-npu", "vllm", "vllm-ascend", "triton", "triton-ascend")},
              "runtime_files": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                                ("/usr/local/Ascend/cann-9.1.0/opp/version.info",
                                 "/usr/local/Ascend/driver/version.info")},
              "probe_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "inputs": [digest_tensor(t) for t in (x_cpu, w_cpu, qw_cpu, kw_cpu,
                                                    positions_cpu, cache_cpu)]}
    if bias_cpu is not None:
        shared["inputs"].append(digest_tensor(bias_cpu))
    baseline = None
    if args.baseline:
        baseline = torch.load(args.baseline, map_location="cpu", weights_only=True)
        if baseline["shared"] != shared or baseline["arm"] != "B0":
            raise RuntimeError("baseline provenance/tensors differ")
    if args.arm == "B1" and baseline is None:
        raise RuntimeError("B1 requires same-seed B0 receipt --baseline")
    fmt = weight_format(args.arm, args.m, args.bias)
    checkpoint("inputs_and_baseline_verified", shared=shared, weight_format=fmt)
    x = torch_npu.npu_format_cast(x_cpu.to("npu:0"), 2)
    w = torch_npu.npu_format_cast(w_cpu.to("npu:0"), 2)
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    if fmt == 29:
        w = torch_npu.npu_format_cast(w, 29)
    torch.npu.synchronize()
    setup_us = (time.perf_counter_ns() - start) / 1000
    gc.collect()
    torch.npu.empty_cache()
    after = torch.npu.memory_allocated()
    load_peak = torch.npu.max_memory_allocated()
    if load_peak > 2 * 1024**3:
        raise RuntimeError("2GiB allocated device budget exceeded at weight load")
    if int(torch_npu.get_npu_format(w)) != fmt:
        raise RuntimeError("requested weight format unavailable")
    qw, kw, cache, positions = [t.to("npu:0") for t in (qw_cpu, kw_cpu, cache_cpu, positions_cpu)]
    bias = bias_cpu.to("npu:0") if bias_cpu is not None else None

    def consume(qkv):
        return consumer(qkv, cache, positions, qw, kw, 2048, 256, 128, 1e-6)

    def chain():
        qkv = F.linear(x, w, bias)
        return (qkv, *consume(qkv))

    def check(values, xc=x_cpu, pc=positions_cpu):
        qkv, q, k, v = values
        if qkv.shape != (args.m, 2560) or qkv.dtype != torch.bfloat16 or not qkv.is_contiguous():
            raise AssertionError("consumer needs contiguous BF16 [M,2560]")
        if int(torch_npu.get_npu_format(qkv)) != 2:
            raise AssertionError("producer output is not physical ND; no hidden repair allowed")
        for t, n in zip((q, k, v), (2048, 256, 256)):
            if t.shape != (args.m, n) or t.dtype != torch.bfloat16 or not t.is_contiguous():
                raise AssertionError("consumer output contract")
        qkv32 = F.linear(xc.float(), w_cpu.float(), bias_cpu.float() if args.bias else None)
        report = {"gemm_fp32": compare(qkv, qkv32, atol=0.05, rtol=0.01, rel_l2_limit=0.01)}
        own_ref = consumer_reference(qkv.cpu(), cache_cpu, pc, qw_cpu, kw_cpu)
        end_ref = consumer_reference(qkv32.bfloat16(), cache_cpu, pc, qw_cpu, kw_cpu)
        for name, actual, own, end in zip(("q", "k", "v"), (q, k, v), own_ref, end_ref):
            report[name] = {"consumer_fp32": compare(actual, own), "chain_fp32": compare(actual, end)}
        compare(v, qkv.cpu()[:, 2304:], atol=0, rtol=0)
        return report

    def capture(fn):
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        capture_context = torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True)
        with capture_context:
            values = fn()
        if args.stage == "stability":
            stream_observation.update(capture_stream=int(capture_context.capture_stream.npu_stream),
                                      replay_caller_stream=int(torch.npu.current_stream().npu_stream))
        graph.replay()
        torch.npu.synchronize()
        return graph, values

    stream_observation = {}
    stability_windows = []
    with torch.inference_mode():
        # Reject incompatible physical output before the stride-blind consumer.
        producer_probe = F.linear(x, w, bias)
        if (producer_probe.shape != (args.m, 2560) or producer_probe.dtype != torch.bfloat16
                or not producer_probe.is_contiguous() or int(torch_npu.get_npu_format(producer_probe)) != 2):
            raise RuntimeError("GEMM output is not consumer-compatible ND; stop without repair")
        del producer_probe
        eager = chain()
        correctness = {"eager": check(eager)}
        checkpoint("eager_oracle_pass", correctness=correctness["eager"])
        graph, values = capture(chain)
        correctness["graph"] = check(values)
        checkpoint("graph_oracle_pass", correctness=correctness["graph"])
        for actual, expected in zip(values, eager):
            compare(actual, expected, atol=0, rtol=0)
        snapshot = [t.cpu().clone() for t in values]
        for _ in range(5):
            graph.replay()
            torch.npu.synchronize()
            for actual, expected in zip(values, snapshot):
                compare(actual, expected, atol=0, rtol=0)
        checkpoint("five_replays_and_eager_exact_pass")
        if baseline:
            correctness["B0_pair"] = [compare(a, b, atol=0 if fmt == 2 else 0.05,
                                               rtol=0 if fmt == 2 else 0.01 if i == 0 else RTOL,
                                               rel_l2_limit=0.01 if i == 0 else None)
                                      for i, (a, b) in enumerate(zip(snapshot, baseline["outputs"]))]
            checkpoint("B0_pair_pass", correctness=correctness["B0_pair"])
        # A static replay must read updated input/position buffers, not stale captures.
        x.copy_(x_cpu.mul(0.5).to("npu:0"))
        positions.copy_((positions_cpu + 1).to("npu:0"))
        graph.replay()
        torch.npu.synchronize()
        correctness["updated_buffers"] = check(values, x_cpu.mul(0.5), positions_cpu + 1)
        checkpoint("updated_buffers_oracle_pass", correctness=correctness["updated_buffers"])
        x.copy_(x_cpu.to("npu:0"))
        positions.copy_(positions_cpu.to("npu:0"))
        graph.replay()
        torch.npu.synchronize()
        # Separate probes are diagnostics, not additive critical-path estimates.
        if args.scope != "chain":
            fixed_qkv = baseline["outputs"][0].to("npu:0") if baseline else eager[0]
            graph, values = capture((lambda: (F.linear(x, w, bias),)) if args.scope == "gemm"
                                    else lambda: consume(fixed_qkv))
        if torch.npu.max_memory_allocated() > 2 * 1024**3:
            raise RuntimeError("2GiB allocated device budget exceeded")
        torch.npu.reset_peak_memory_stats()
        timings = []
        timing_observation = None
        profile_observation = None
        if args.stage == "stability":
            from stability_state import snapshot as state_snapshot, validate as validate_state
            weight_ptr = w.data_ptr()
            for window in range(1, args.windows + 1):
                def save_state(phase):
                    state = state_snapshot(args.physical_card)
                    (out / f"window{window}_{phase}.local.json").write_text(json.dumps(state, indent=2))
                    validate_state(state, args.device_identity_sha)

                save_state("before")
                for _ in range(10):
                    graph.replay()
                torch.npu.synchronize()
                window_trials = []
                for trial in range(1, 6):
                    stream = int(torch.npu.current_stream().npu_stream)
                    if stream != stream_observation["replay_caller_stream"]:
                        raise RuntimeError("replay caller stream changed")
                    a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    wall = time.perf_counter_ns()
                    a.record()
                    enqueue_start = time.perf_counter_ns()
                    for _ in range(100):
                        graph.replay()
                    enqueue_end = time.perf_counter_ns()
                    b.record()
                    sync_start = time.perf_counter_ns()
                    torch.npu.synchronize()
                    sync_end = time.perf_counter_ns()
                    event_us = a.elapsed_time(b) * 1000 / 100
                    wall_end = time.perf_counter_ns()
                    row = {"event_us": event_us, "wall_us": (wall_end - wall) / 1000 / 100,
                           "enqueue_us": (enqueue_end - enqueue_start) / 1000 / 100,
                           "synchronize_us": (sync_end - sync_start) / 1000 / 100,
                           "start_record_us": (enqueue_start - wall) / 1000 / 100,
                           "end_record_us": (sync_start - enqueue_end) / 1000 / 100,
                           "event_query_us": (wall_end - sync_end) / 1000 / 100}
                    window_trials.append(row)
                    checkpoint("stability_trial_completed", window=window, trial=trial, **row)
                save_state("after")
                oracle = check(values)
                for actual, expected in zip(values, snapshot):
                    compare(actual, expected, atol=0, rtol=0)
                observation = {"weight_pointer_unchanged": w.data_ptr() == weight_ptr,
                               "weight_format_after": int(torch_npu.get_npu_format(w)),
                               "allocated": torch.npu.memory_allocated(),
                               "peak_allocated": torch.npu.max_memory_allocated()}
                if (not observation["weight_pointer_unchanged"] or observation["weight_format_after"] != 2
                        or observation["peak_allocated"] > 2 * 1024**3):
                    raise RuntimeError("stability identity/format/allocation budget failure")
                stability_windows.append({"window": window, "trials": window_trials,
                                          "post_window_oracle": oracle, "observation": observation})
                checkpoint("stability_window_oracle_pass", window=window, observation=observation)
        if args.stage == "timing":
            weight_ptr = w.data_ptr()
            allocated_before_timing = torch.npu.memory_allocated()
            for _ in range(10):
                graph.replay()
            torch.npu.synchronize()
            for _ in range(5):
                a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                wall = time.perf_counter_ns()
                a.record()
                for _ in range(100):
                    graph.replay()
                b.record()
                torch.npu.synchronize()
                timings.append({"event_us": a.elapsed_time(b) * 1000 / 100,
                                "wall_us": (time.perf_counter_ns() - wall) / 1000 / 100})
                checkpoint("timing_trial_completed", trial=len(timings), **timings[-1])
            timing_observation = {
                "warmup_replays": 10, "trials": 5, "replays_per_trial": 100,
                "weight_pointer_unchanged": w.data_ptr() == weight_ptr,
                "weight_format_after": int(torch_npu.get_npu_format(w)),
                "allocated_before": allocated_before_timing,
                "allocated_after": torch.npu.memory_allocated(),
                "peak_allocated": torch.npu.max_memory_allocated(),
            }
            if (not timing_observation["weight_pointer_unchanged"]
                    or timing_observation["weight_format_after"] != fmt
                    or timing_observation["peak_allocated"] > 2 * 1024**3):
                raise RuntimeError("timing changed weight identity/format or exceeded allocation budget")
            if args.scope == "chain":
                correctness["post_timing"] = check(values)
                for actual, expected in zip(values, snapshot):
                    compare(actual, expected, atol=0, rtol=0)
                checkpoint("post_timing_oracle_and_exact_pass", correctness=correctness["post_timing"])
        if args.stage == "profile":
            from torch_npu.profiler import _ExperimentalConfig, ProfilerLevel, AiCMetrics
            weight_ptr = w.data_ptr()
            allocated_before_profile = torch.npu.memory_allocated()
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out / "profiler")),
                record_shapes=True, profile_memory=True,
                experimental_config=_ExperimentalConfig(
                    profiler_level=ProfilerLevel.Level1, aic_metrics=AiCMetrics.AiCoreNone,
                    data_simplification=False, op_attr=True,
                ),
            ):
                for i in range(5):
                    with torch.profiler.record_function(f"M0B_GRAPH_REPLAY_{i}"):
                        graph.replay()
                torch.npu.synchronize()
            profile_observation = {
                "replays": 5, "weight_pointer_unchanged": w.data_ptr() == weight_ptr,
                "weight_format_after": int(torch_npu.get_npu_format(w)),
                "allocated_before": allocated_before_profile,
                "allocated_after": torch.npu.memory_allocated(),
                "peak_allocated": torch.npu.max_memory_allocated(),
            }
            checkpoint("profile_completed", observation=profile_observation)
            if (not profile_observation["weight_pointer_unchanged"]
                    or profile_observation["weight_format_after"] != fmt
                    or profile_observation["peak_allocated"] > 2 * 1024**3):
                raise RuntimeError("profile changed weight identity/format or exceeded allocation budget")
            if args.scope == "chain":
                correctness["post_profile"] = check(values)
                for actual, expected in zip(values, snapshot):
                    compare(actual, expected, atol=0, rtol=0)
                checkpoint("post_profile_oracle_and_exact_pass", correctness=correctness["post_profile"])
        torch.save({"shared": shared, "arm": args.arm, "outputs": snapshot}, out / "outputs.pt")
        return {"status": "DEVICE_CHECKS_PASS_NOT_DISPATCH_ADMISSION", "shared": shared,
                "mode": args.mode, "formal_approval": manifest["approval"]["approved_for_hardware"],
                "arm": args.arm, "scope": args.scope, "stage": args.stage, "weight_format": fmt,
                "load_cast_wall_us": setup_us, "allocated_before_cast": before,
                "allocated_after_release": after, "load_peak_allocated": load_peak,
                "memory_allocated": torch.npu.memory_allocated(),
                "memory_reserved": torch.npu.memory_reserved(), "peak_allocated": torch.npu.max_memory_allocated(),
                "layouts": [{"shape": list(t.shape), "stride": list(t.stride()),
                             "dtype": str(t.dtype), "format": int(torch_npu.get_npu_format(t))}
                            for t in (x, w, *eager)],
                "correctness": correctness, "trials": timings,
                "stability_windows": stability_windows, "stream_observation": stream_observation,
                "timing_observation": timing_observation,
                "profile_observation": profile_observation,
                "median_us": statistics.median(t["event_us"] for t in timings) if timings else None,
                "median_wall_us": statistics.median(t["wall_us"] for t in timings) if timings else None,
                "dispatch_and_hot_conversion_audit": "REQUIRES_TRACE_REVIEW"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("plan", "correctness", "timing", "profile", "stability"), default="plan")
    p.add_argument("--windows", type=int, choices=(1, 3), default=1)
    p.add_argument("--physical-card", type=int)
    p.add_argument("--device-identity-sha")
    p.add_argument("--mode", choices=("FORMAL", "DIAGNOSTIC_ONLY"), default="FORMAL")
    p.add_argument("--arm", choices=("B0", "B1"), default="B0")
    p.add_argument("--m", type=int, choices=M_VALUES, default=1)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--seed", type=int, choices=range(10), default=0)
    p.add_argument("--scope", choices=("chain", "gemm", "consumer"), default="chain")
    p.add_argument("--baseline", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.stage == "plan":
        print(json.dumps(plan(), indent=2))
        return
    import yaml
    manifest = yaml.safe_load((ROOT / "docs/M0_MANIFEST.yaml").read_text())
    require_execution_permission(manifest, args)
    if args.bias and args.m != 1:
        p.error("only M=1 bias fallback is preregistered")
    if args.out is None or not args.out.resolve().is_relative_to(ROOT / "m0_receipts"):
        p.error("--out must be a new local m0_receipts subdirectory")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        result = hardware(args, manifest, args.out)
    except Exception as exc:
        (args.out / "FAILED.local.json").write_text(json.dumps({"status": "FAILED", "error": repr(exc)}))
        raise
    (args.out / "result.local.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
