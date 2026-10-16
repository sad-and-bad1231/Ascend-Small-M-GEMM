"""Read-only, outside-window device/host snapshots. Raw output stays local."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=5)
    return {"command": args, "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def identity(board):
    fields = dict(line.strip().split(":", 1) for line in board.splitlines() if ":" in line)
    selected = {k.strip(): v.strip() for k, v in fields.items()
                if k.strip() in ("NPU ID", "Chip ID", "Chip Name", "VDie ID", "PCIe Bus Info")}
    if len(selected) != 5 or selected["Chip Name"] != "910B2":
        raise RuntimeError("missing physical identity or unexpected device")
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def snapshot(card):
    started = time.time_ns()
    queries = {kind: command(["npu-smi", "info", "-t", kind, "-i", str(card), "-c", "0"])
               for kind in ("board", "sensors", "power", "usages")}
    queries["overview"] = command(["npu-smi", "info"])
    queries["mapping"] = command(["npu-smi", "info", "-m"])
    frequency = {}
    try:
        lib = ctypes.CDLL("/usr/local/Ascend/driver/lib64/driver/libdcmi.so")
        lib.dcmi_init.restype = ctypes.c_int
        init_rc = lib.dcmi_init()
        frequency["init_rc"] = init_rc
        if init_rc == 0:
            fn = lib.dcmi_get_device_frequency
            fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)]
            fn.restype = ctypes.c_int
            # Read-only enum values from installed dcmi_interface_api.h.
            for name, kind in (("aicore_current", 7), ("aicore_max", 9),
                               ("vector_current", 12), ("hbm", 6)):
                value = ctypes.c_uint()
                rc = fn(card, 0, kind, ctypes.byref(value))
                frequency[name] = {"rc": rc, "value": value.value if rc == 0 else None}
    except (OSError, AttributeError) as exc:
        frequency["unavailable"] = repr(exc)
    files = ("/proc/stat", "/proc/loadavg", "/proc/self/status", "/proc/self/schedstat",
             "/proc/pressure/cpu", "/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu.max",
             "/sys/fs/cgroup/cpuset.cpus.effective", "/sys/fs/cgroup/cpu.pressure")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"started_ns": started, "ended_ns": time.time_ns(), "pid": os.getpid(),
            "identity_sha256": (identity(queries["board"]["stdout"])
                                if queries["board"]["exit_code"] == 0 else None), "queries": queries,
            "frequency_dcmi": frequency, "affinity": sorted(os.sched_getaffinity(0)),
            "host_files": {p: Path(p).read_text() for p in files if Path(p).exists()},
            "process_cpu_seconds": {"user": usage.ru_utime, "system": usage.ru_stime},
            "context_switches": {"voluntary": usage.ru_nvcsw, "involuntary": usage.ru_nivcsw}}


def validate(state, expected):
    if any(q["exit_code"] for q in state["queries"].values()):
        raise RuntimeError("required device snapshot query failed; raw query receipts retained")
    if state["identity_sha256"] != expected:
        raise RuntimeError("physical device identity changed")
    # Mapping proves the selected physical chip is logical device0.
    board = state["queries"]["board"]["stdout"]
    card = next(line.split(":", 1)[1].strip() for line in board.splitlines() if "NPU ID" in line)
    mappings = [line.split() for line in state["queries"]["mapping"]["stdout"].splitlines()]
    if not any(row[:3] == [card, "0", "0"] for row in mappings):
        raise RuntimeError("selected physical device is not logical0")
    overview = state["queries"]["overview"]["stdout"]
    if "OK" not in overview:
        raise RuntimeError("device not healthy")
    process_section = overview.split("Process id", 1)[-1]
    for line in process_section.splitlines():
        row = line.strip("| ").split()
        if len(row) >= 5 and row[0].isdigit() and row[1].isdigit():
            if str(os.getpid()) not in (row[2], row[-1]):
                raise RuntimeError("unexpected NPU process; stop without terminating it")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.card), indent=2))
