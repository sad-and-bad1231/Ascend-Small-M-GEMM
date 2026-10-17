"""Final small paired screen, reusing the existing runtime, capture and oracle."""

import gc
import json
import math
import statistics
import time

ORDER = ("B0", "shadow", "shadow", "B0", "B0", "B1", "B1", "B0",
         "B0", "shadow", "shadow", "B0")


def assess(rows):
    if len(rows) != 60 or any((r["cycle"], r["slot"], r["arm"]) != (c, i, arm)
                             for r, (c, i, arm) in zip(rows, ((c, i, arm) for c in range(5) for i, arm in enumerate(ORDER)))):
        raise ValueError("missing/reordered trials; no filtering allowed")
    if any(not math.isfinite(r[k]) or r[k] <= 0 for r in rows for k in ("event_us", "wall_us")):
        raise ValueError("nonfinite or nonpositive trial")
    result, invalid = {}, False
    for metric in ("event_us", "wall_us"):
        shadow, gain, bracket = [], [], []
        for cycle in range(5):
            values = [r[metric] for r in rows if r["cycle"] == cycle]
            for start in (0, 4, 8):
                reference = (values[start] + values[start + 3]) / 2
                middle = (values[start + 1] + values[start + 2]) / 2
                bracket.append(abs(values[start + 3] - values[start]) / values[start] * 100)
                (gain if start == 4 else shadow).append((reference - middle) / reference * 100)
        first = statistics.median(r[metric] for r in rows if r["cycle"] == 0 and r["arm"] == "B0")
        last = statistics.median(r[metric] for r in rows if r["cycle"] == 4 and r["arm"] == "B0")
        drift = abs(last - first) / first * 100
        failed = any(abs(v) >= 2 for v in shadow) or any(v >= 2 for v in bracket) or drift >= 2
        invalid |= failed
        result[metric] = {"shadow_signed_reduction_percent_all10_pairs": shadow,
                          "B0_bracket_absolute_drift_percent_all15": bracket,
                          "B0_first_last_cycle_median_drift_percent": drift,
                          "B1_paired_reduction_percent_all5": gain,
                          "B1_paired_reduction_median_percent_raw": statistics.median(gain),
                          "control_gate_failed": failed}
    return {"status": "INVALID_CONTROL" if invalid else "VALID_SCREEN_ONLY",
            "metrics": result, "accepted_chain_gain_percent": None if invalid else
            result["event_us"]["B1_paired_reduction_median_percent_raw"], "formal_approved": False}


def run_screen(args, out, shared, baseline, x, w, w_cpu, positions, positions_cpu,
               x_cpu, consume, check, capture, checkpoint):
    import torch
    import torch_npu
    import torch.nn.functional as F
    from qkv_probe import compare, RTOL
    from stability_state import snapshot, validate

    with torch.inference_mode():
        weights = {"B0": w, "shadow": torch_npu.npu_format_cast(w_cpu.to("npu:0"), 2),
                   "B1": torch_npu.npu_format_cast(w_cpu.to("npu:0"), 29)}
        gc.collect()
        torch.npu.empty_cache()
        pointers = {k: t.data_ptr() for k, t in weights.items()}
        if len(set(pointers.values())) != 3:
            raise RuntimeError("weights are not three independent allocations")
        graphs, outputs, snapshots, oracles = {}, {}, {}, {}
        for arm, weight in weights.items():
            def chain(weight=weight):
                qkv = F.linear(x, weight)
                return (qkv, *consume(qkv))
            producer = F.linear(x, weight)
            if int(torch_npu.get_npu_format(producer)) != 2 or not producer.is_contiguous():
                raise RuntimeError("producer not ND compatible; no repair")
            del producer
            eager = chain()
            oracles[arm] = {"eager": check(eager)}
            graph, values = capture(chain)
            oracles[arm]["graph"] = check(values)
            for a, b in zip(values, eager):
                compare(a, b, atol=0, rtol=0)
            saved = [t.cpu().clone() for t in values]
            for _ in range(5):
                graph.replay()
                torch.npu.synchronize()
                for a, b in zip(values, saved):
                    compare(a, b, atol=0, rtol=0)
            x.copy_(x_cpu.mul(0.5).to("npu:0"))
            positions.copy_((positions_cpu + 1).to("npu:0"))
            graph.replay()
            torch.npu.synchronize()
            oracles[arm]["changed_buffers"] = check(values, x_cpu.mul(0.5), positions_cpu + 1)
            x.copy_(x_cpu.to("npu:0"))
            positions.copy_(positions_cpu.to("npu:0"))
            graph.replay()
            torch.npu.synchronize()
            graphs[arm], outputs[arm], snapshots[arm] = graph, values, saved
            if arm != "B0":
                oracles[arm]["B0_pair"] = [compare(a, b, atol=0 if arm == "shadow" else 0.05,
                    rtol=0 if arm == "shadow" else 0.01 if i == 0 else RTOL,
                    rel_l2_limit=0.01 if i == 0 else None)
                    for i, (a, b) in enumerate(zip(saved, snapshots["B0"]))]
            checkpoint("screen_graph_full_oracle_pass", arm=arm, oracle=oracles[arm])
        if baseline:
            oracles["cross_process_B0_pair"] = [compare(a, b, atol=0, rtol=0)
                                                 for a, b in zip(snapshots["B0"], baseline["outputs"])]
        if len({v[0].data_ptr() for v in outputs.values()}) != 3:
            raise RuntimeError("graphs unexpectedly alias QKV output")
        before = snapshot(args.physical_card)
        (out / "screen_before.local.json").write_text(json.dumps(before, indent=2))
        validate(before, args.device_identity_sha)
        if torch.npu.max_memory_allocated() > 2 * 1024**3:
            raise RuntimeError("2GiB load/capture budget exceeded")
        a, b = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        for _ in range(3):
            a.record()
            b.record()
            torch.npu.synchronize()
            a.elapsed_time(b)
        rows = []
        stream = int(torch.npu.current_stream().npu_stream)
        for cycle in range(5):
            for slot, arm in enumerate(ORDER):
                if int(torch.npu.current_stream().npu_stream) != stream:
                    raise RuntimeError("measurement caller stream changed")
                graph = graphs[arm]
                for _ in range(10):
                    graph.replay()
                torch.npu.synchronize()
                wall = time.perf_counter_ns()
                a.record()
                for _ in range(100):
                    graph.replay()
                b.record()
                torch.npu.synchronize()
                row = {"cycle": cycle, "slot": slot, "arm": arm,
                       "event_us": a.elapsed_time(b) * 1000 / 100,
                       "wall_us": (time.perf_counter_ns() - wall) / 1000 / 100}
                rows.append(row)
                checkpoint("screen_trial", **row)
        for arm, values in outputs.items():
            oracles[arm]["post_screen"] = check(values)
            for actual, expected in zip(values, snapshots[arm]):
                compare(actual, expected, atol=0, rtol=0)
        after = snapshot(args.physical_card)
        (out / "screen_after.local.json").write_text(json.dumps(after, indent=2))
        validate(after, args.device_identity_sha)
        formats = {k: int(torch_npu.get_npu_format(t)) for k, t in weights.items()}
        peak = torch.npu.max_memory_allocated()
        if (formats != {"B0": 2, "shadow": 2, "B1": 29} or peak > 2 * 1024**3
                or any(t.data_ptr() != pointers[k] for k, t in weights.items())):
            raise RuntimeError("format/pointer/allocated budget changed")
        torch.save({"shared": shared, "arm": "B0", "outputs": snapshots["B0"]}, out / "outputs.pt")
        return {"shared": shared, "mode": args.mode, "formal_approved": False, "m": args.m,
                "oracles": oracles, "trials": rows, "assessment": assess(rows),
                "three_weights_co_resident": True, "independent_graphs_and_outputs": 3,
                "weight_formats": formats, "allocated_bytes": torch.npu.memory_allocated(),
                "peak_allocated_bytes": peak, "event_api_prewarm_pairs": 3,
                "weight_pointers_stable": True}
