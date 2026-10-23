"""PR14 QKV-only load-time policy, adapted to the current isolated runner."""
import json
import os
import time
from pathlib import Path


def apply_policy(runner):
    import torch
    import torch_npu
    from vllm_ascend.ascend_config import get_ascend_config

    mode = os.environ["VLLM_ASCEND_QKV_NZ_MODE"]
    assert mode in ("0", "1")
    assert get_ascend_config().weight_nz_mode != 2
    assert runner.vllm_config.quant_config is None
    assert runner.vllm_config.scheduler_config.max_num_seqs == 8
    assert torch.npu.device_count() == 4
    converted, inventory = 0, []
    torch.npu.synchronize()
    start = time.perf_counter()
    for name, param in runner.model.named_parameters():
        if not name.endswith(".qkv_proj.weight"):
            continue
        parent = runner.model.get_submodule(name[:-len(".weight")])
        assert parent.bias is None
        assert tuple(param.shape) == (2560, 5120)
        assert param.dtype == torch.bfloat16
        before = torch_npu.get_npu_format(param)
        assert before == 2, (name, before)
        if mode == "1":
            param.data = torch_npu.npu_format_cast(param.data, 29)
            converted += 1
        inventory.append({"name": name, "shape": list(param.shape),
                          "before": before, "after": torch_npu.get_npu_format(param)})
    torch.npu.synchronize()
    assert len(inventory) == 64
    assert converted == (64 if mode == "1" else 0)
    rank = runner.parallel_config.rank
    result = {"rank": rank, "mode": mode, "converted": converted,
              "conversion_wall_s": time.perf_counter() - start,
              "weight_nz_mode": get_ascend_config().weight_nz_mode,
              "allow_internal_format": "NO_PUBLIC_GETTER_FORMATS_VERIFIED_DIRECTLY",
              "allocated_bytes_after_load": torch.npu.memory_allocated(),
              "peak_allocated_bytes_after_load": torch.npu.max_memory_allocated(),
              "qkv_weights": inventory}
    out = Path(os.environ["M0_ARM_OUT"])
    (out / f"rank{rank}_policy.local.json").write_text(json.dumps(result, indent=2))
    print(f"M0_QKV_POLICY rank={rank} mode={mode} converted={converted}", flush=True)


def memory_snapshot(worker):
    import torch
    return {"rank": worker.rank, "allocated_bytes": torch.npu.memory_allocated(),
            "reserved_bytes": torch.npu.memory_reserved(),
            "peak_allocated_bytes": torch.npu.max_memory_allocated()}
