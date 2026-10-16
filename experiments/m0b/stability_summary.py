"""Small stdlib-only reducer for the authorized four-process B0 diagnostic."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = ("event_us", "wall_us", "enqueue_us", "synchronize_us",
           "start_record_us", "end_record_us", "event_query_us")
RUNS = ("same_process", "cross_process1", "cross_process2", "cross_process3")


def variation(values):
    if len(values) != 3 or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("need three positive window/process medians")
    pairs = [{"earlier": i + 1, "later": j + 1,
              "absolute_drift_percent": abs(values[j] - values[i]) / values[i] * 100}
             for i in range(3) for j in range(i + 1, 3)]
    return {"medians_us": values, "endpoint_drift_percent": pairs[1]["absolute_drift_percent"],
            "spread_over_first_percent": (max(values) - min(values)) / values[0] * 100,
            "pairwise": pairs,
            "blocks_ABA": any(p["absolute_drift_percent"] >= 2 for p in pairs)}


def fields(text):
    return {k.strip(): v.strip() for k, v in
            (line.split(":", 1) for line in text.splitlines() if ":" in line)}


def public_state(state):
    host = state["host_files"]
    return {"frequency_driver_values": state["frequency_dcmi"],
            "sensors": fields(state["queries"]["sensors"]["stdout"]),
            "power": fields(state["queries"]["power"]["stdout"]),
            "utilization": fields(state["queries"]["usages"]["stdout"]),
            "affinity_cpu_count": len(state["affinity"]),
            "loadavg_1m": float(host["/proc/loadavg"].split()[0]),
            "cgroup_cpu_stat": host.get("/sys/fs/cgroup/cpu.stat"),
            "cgroup_cpu_max": host.get("/sys/fs/cgroup/cpu.max"),
            "cgroup_cpu_pressure": host.get("/sys/fs/cgroup/cpu.pressure"),
            "host_cpu_pressure": host.get("/proc/pressure/cpu"),
            "process_cpu_seconds": state["process_cpu_seconds"],
            "context_switches": state["context_switches"]}


def summarize(root):
    root = Path(root)
    initial = json.loads((root / "device_initial.local.json").read_text())
    runs, shared = [], None
    for name, count in zip(RUNS, (3, 1, 1, 1)):
        path = root / name / "result.local.json"
        record = json.loads(path.read_text())
        if ((root / (name + ".exit.local.txt")).read_text().strip() != "0"
                or record["stage"] != "stability" or record["arm"] != "B0"
                or record["mode"] != "DIAGNOSTIC_ONLY" or record["formal_approval"] is not False
                or record["scope"] != "chain" or record["weight_format"] != 2
                or record["shared"]["m"] != 1 or record["shared"]["seed"] != 0
                or record["shared"]["bias"] or record["profile_observation"] is not None
                or record["load_peak_allocated"] > 2 * 1024**3):
            raise ValueError("failed or outside authorized stability scope")
        if shared is not None and record["shared"] != shared:
            raise ValueError("input/source/runtime mismatch")
        shared = record["shared"]
        windows = record["stability_windows"]
        if len(windows) != count:
            raise ValueError("missing or extra windows")
        for window in windows:
            trials = window["trials"]
            obs = window["observation"]
            if (len(trials) != 5 or not window["post_window_oracle"]
                    or not obs["weight_pointer_unchanged"] or obs["weight_format_after"] != 2
                    or obs["peak_allocated"] > 2 * 1024**3):
                raise ValueError("trial/oracle/resource failure")
            for trial in trials:
                if not all(math.isfinite(trial[k]) and trial[k] >= 0 for k in METRICS):
                    raise ValueError("nonfinite or negative timing")
                parts = sum(trial[k] for k in METRICS[2:])
                if not math.isclose(parts, trial["wall_us"], abs_tol=1e-8):
                    raise ValueError("Wall decomposition mismatch")
            states = {}
            for phase in ("before", "after"):
                state = json.loads((root / name / f"window{window['window']}_{phase}.local.json").read_text())
                if (state["identity_sha256"] != initial["identity_sha256"]
                        or any(q["exit_code"] for q in state["queries"].values())):
                    raise ValueError("identity/query mismatch")
                states[phase] = public_state(state)
            window["states"] = states
            window["medians_us"] = {k: statistics.median(t[k] for t in trials) for k in METRICS}
            window["oracle"] = "PASS_FULL_AND_EXACT"
            del window["post_window_oracle"]
        streams = record["stream_observation"]
        runs.append({"run": name, "windows": windows, "exit_code": 0,
                     "capture_stream_differs_from_replay_caller":
                     streams["capture_stream"] != streams["replay_caller_stream"],
                     "B0_pair_errors": record["correctness"].get("B0_pair"),
                     "raw_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    same = {k: variation([w["medians_us"][k] for w in runs[0]["windows"]]) for k in METRICS[:2]}
    cross = {k: variation([r["windows"][0]["medians_us"][k] for r in runs[1:]]) for k in METRICS[:2]}
    return {"batch": "m0b_stability_new_allocation_20261016", "formal_approved": False,
            "executed_probe_sha256": shared["probe_sha"], "scope": "B0_only_M1_seed0_chain",
            "matched_input_source_runtime": True, "same_physical_device_all_windows": True,
            "same_process": same, "cross_process": cross, "runs": runs,
            "ABA_readiness": "BLOCKED_VARIATION" if any(v["blocks_ABA"] for v in (*same.values(), *cross.values()))
            else "SCREEN_PASS_DESCRIPTIVE_ONLY_NOT_ABA_AUTHORIZATION",
            "root_cause": "UNKNOWN", "profiler": "NOT_RUN"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
