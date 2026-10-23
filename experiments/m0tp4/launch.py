#!/usr/bin/env python3
"""Own one bounded TP4 process session; clean up only its descendants."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def launch(args):
    root = Path(__file__).resolve().parents[2]
    raw = args.raw.resolve()
    assert len((raw / "checkpoint_weights.sha256").read_text().splitlines()) == 17
    out = raw / args.label
    assert not out.exists(), "never overwrite an attempt"
    before = subprocess.check_output(["npu-smi", "info"], text=True)
    assert before.count("No running processes found") == 4, "NPU occupied"
    (raw / f"{args.label}.before.local.txt").write_text(before)
    env = os.environ.copy()
    assert not env.get("ASCEND_RT_VISIBLE_DEVICES"), "unexpected inherited remapping"
    env.update(M0_OWNER_TP4_AUTHORIZED="1", VLLM_ASCEND_QKV_NZ_MODE="1" if args.arm == "B1" else "0",
               OMP_NUM_THREADS="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    env["PYTHONPATH"] = os.pathsep.join([
        str(raw / "runtime/plugin"), str(root / "experiments/m0tp4"),
        "/vllm-workspace/vllm", env.get("PYTHONPATH", "")])
    command = [sys.executable, str(root / "experiments/m0tp4/run_arm.py"),
               "--arm", args.arm, "--out", str(out), "--model", args.model,
               "--prompts", str(raw / "trace_fixed_120.jsonl"), "--pr14", args.pr14]
    if args.reference:
        command += ["--reference", str(raw / args.reference)]
    if args.profile:
        command += ["--profile"]
    files = [root / "experiments/m0tp4" / name for name in
             ("run_arm.py", "qkv_policy.py", "launch.py")]
    files += [raw / "runtime/plugin/vllm_ascend/worker/model_runner_v1.py",
              raw / "runtime/plugin/vllm_ascend/worker/worker.py",
              raw / "runtime/plugin/vllm_ascend/profiler/torch_npu_profiler.py",
              raw / "runtime/plugin/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py",
              raw / "runtime_identity.local.json",
              raw / "trace_fixed_120.jsonl", raw / "checkpoint_weights.sha256"]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    record = {"command": command, "formal_approved": False, "timeout_s": 1200,
              "reproduction": "CLEAN_PROSPECTIVE_CURRENT_RUNTIME", "source_sha256": source_hashes}
    path = raw / f"{args.label}.launch.local.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"START {args.label}", flush=True)
    start = time.monotonic()
    with (raw / f"{args.label}.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    try:
        code = process.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        code = "TIMEOUT"
    finally:
        # This process group was created above, never a broad Python/NPU kill.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        for _ in range(20):
            time.sleep(2)
            after = subprocess.check_output(["npu-smi", "info"], text=True)
            if after.count("No running processes found") == 4:
                break
        else:
            # Reap only our group if a child survived the driver's hard exit.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(5)
            after = subprocess.check_output(["npu-smi", "info"], text=True)
        (raw / f"{args.label}.after.local.txt").write_text(after)
    record.update(exit_code=code, wall_s=time.monotonic() - start,
                  hbm_processes_released=after.count("No running processes found") == 4,
                  completed=(out / "inventory.local.json").is_file())
    path.write_text(json.dumps(record, indent=2))
    print(f"END {args.label}: {record['exit_code']}, complete={record['completed']}", flush=True)
    assert record["hbm_processes_released"], "own session cleanup failed"
    assert code == 0 and record["completed"], "arm failed; inspect preserved log"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--arm", choices=["B0", "B1"], required=True)
    parser.add_argument("--reference")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--model", default="/data/shared_upload/models/Qwen/Qwen3-32B")
    parser.add_argument("--pr14", default="/root/kmx/repos/research/ascend-small-m-gemm-pr14-m0a")
    launch(parser.parse_args())
