#!/usr/bin/env python3
"""One eight-request, four-token teacher-forced TP4 numeric diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def run(args):
    assert os.environ["M0_OWNER_TP4_AUTHORIZED"] == "1"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ["M0_ARM_OUT"] = str(out)
    input_path = out.parent / "frozen_inputs.local.json"
    os.environ["M0_NUMERIC_INPUT"] = str(input_path)
    os.environ["M0_NUMERIC_DIAGNOSTIC"] = "1"
    inputs = json.loads(input_path.read_text())
    from vllm import LLM, SamplingParams
    config = dict(model=args.model, dtype="bfloat16", tensor_parallel_size=4,
                  max_model_len=4096, max_num_seqs=8, gpu_memory_utilization=0.9,
                  max_num_batched_tokens=8192, enable_chunked_prefill=True,
                  enable_prefix_caching=True, async_scheduling=not args.sync_scheduler,
                  disable_log_stats=False, seed=0, enforce_eager=args.eager,
                  compilation_config={"cudagraph_mode": "NONE" if args.eager else "FULL_DECODE_ONLY",
                                      "cudagraph_capture_sizes": [1, 2, 4, 8],
                                      "max_cudagraph_capture_size": 8})
    if args.profile:
        config["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": str(out / "trace"),
            "ignore_frontend": True, "torch_profiler_with_stack": False, "max_iterations": 2}
    (out / "identity.local.json").write_text(json.dumps({"arm": args.arm, "config": config,
        "formal_approved": False, "diagnostic_only": True,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}, indent=2))
    llm = LLM(**config)
    assert all(llm.collective_rpc("m0_numeric_arm"))
    llm.sleep(level=0)
    sampling = SamplingParams(max_tokens=4, temperature=0, ignore_eos=True, seed=0)
    llm.enqueue([{"prompt_token_ids": ids} for ids in inputs["prompt_ids"]], sampling, use_tqdm=False)
    if args.profile:
        llm.start_profile(profile_prefix="numeric")
    llm.wake_up(tags=["scheduling"])
    outputs = llm.wait_for_completion(use_tqdm=False)
    if args.profile:
        llm.stop_profile()
    assert len(outputs) == 8
    for index, output in enumerate(outputs):
        assert output.prompt_token_ids == inputs["prompt_ids"][index]
        assert output.finished and len(output.outputs[0].token_ids) == 4
        assert list(output.outputs[0].token_ids[:3]) == inputs["forced_prefix"][index]
    memory = llm.collective_rpc("m0_numeric_dump")
    (out / "outputs.local.json").write_text(json.dumps([list(x.outputs[0].token_ids) for x in outputs]))
    (out / "inventory.local.json").write_text(json.dumps({"memory": memory, "requests": 8,
        "tokens": 32, "formal_approved": False, "performance_measurement": False}, indent=2))
    print("M0_NUMERIC_COMPLETE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("arm", "model", "prompts", "pr14"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--sync-scheduler", action="store_true")
    run(parser.parse_args())
