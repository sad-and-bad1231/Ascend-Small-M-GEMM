"""Post-forward-only numeric custody; no probe inside captured model graphs."""
import json
import os
from pathlib import Path

_armed = False
_steps = []
_local = []
_order = None


def arm():
    global _armed, _order
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    original = LogitsProcessor._gather_logits

    def gather(self, logits):
        if _armed:
            _local.append((logits.detach().clone(), bool(self.use_all_gather)))
        return original(self, logits)

    LogitsProcessor._gather_logits = gather
    _steps.clear()
    _local.clear()
    _order = None
    _armed = True
    return True


def observe(runner, logits, hidden, input_ids, positions, scheduled, mode, descriptor):
    global _order
    if not _armed:
        return logits
    import torch
    from vllm.distributed.parallel_state import get_tp_group
    prefix = json.loads(Path(os.environ["M0_NUMERIC_INPUT"]).read_text())["forced_prefix"]
    ids = list(runner.input_batch.req_ids)
    if _order is None:
        _order = ids
    assert len(ids) == 8 and set(ids) == set(_order)
    assert logits.shape[0] == 8 and logits.shape[1] >= 151936
    assert len(_steps) < 8, "bounded numeric forward budget"
    group = get_tp_group()
    local, all_gather = _local[-1] if _local else (None, None)
    metadata = {"step": len(_steps), "rank": runner.parallel_config.rank,
                "request_order": [_order.index(x) for x in ids],
                "scheduled_tokens": scheduled.tolist(), "M_padded": descriptor.num_tokens,
                "graph_mode": str(mode), "async_scheduling": runner.use_async_scheduling,
                "num_computed_tokens_cpu": runner.input_batch.num_computed_tokens_cpu[:8].tolist(),
                "TP_world_size": group.world_size,
                "backend": torch.distributed.get_backend(group.device_group),
                "logits_all_gather": all_gather}
    _steps.append({"metadata": metadata, "input_ids": input_ids.detach().clone(),
                   "positions": positions.detach().clone(), "hidden": hidden.detach().clone(),
                   "logits": logits.detach().clone(), "local_logits": local})
    if len(_steps) <= 3:
        forced = torch.full_like(logits, -float("inf"))
        tokens = torch.tensor([prefix[_order.index(x)][len(_steps)-1] for x in ids], device=logits.device)
        forced.scatter_(1, tokens[:, None], 0)
        return forced
    return logits


def dump(worker):
    global _armed
    import torch
    _armed = False
    torch.npu.synchronize()
    records = [{key: (value.cpu() if isinstance(value, torch.Tensor) else value)
                for key, value in step.items()} for step in _steps]
    out = Path(os.environ["M0_ARM_OUT"])
    torch.save(records, out / f"rank{worker.rank}_numeric.local.pt")
    memory = {"rank": worker.rank, "steps": len(records),
              "peak_allocated_bytes": torch.npu.max_memory_allocated()}
    assert memory["peak_allocated_bytes"] <= 58 * 2**30
    return memory
