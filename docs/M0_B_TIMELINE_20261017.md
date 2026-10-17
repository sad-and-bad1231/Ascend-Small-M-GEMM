# M0-B minimal device timeline attribution

**Observed:** Consumer duration changes substantially across these three B0
processes; handoff gap stays tiny. Device Graph envelope follows unprofiled Event
variation. Why Consumer duration changes remains **UNKNOWN**; A/B/A stays blocked.

Preregistered2026-10-17, base `39d5b9f20706509ad89ca92e0f617c0f88e8aaf9`.
Owner-authorized `DIAGNOSTIC_ONLY`; `formal_approved=false`, A/B/A blocked.
Current allocated single910B2 identity is frozen for this independent batch;
no cross-card comparison with previous absolute latency values.

Fixed order: process1, process2, process3, fresh processes, B0 ND-only,
BF16 M=1/K5120/N2560, seed0, no bias, same existing consumer/Graph.
Each process: full original oracle ->10 warmup +5x100 unprofiled Event/Wall ->
full/exact oracle -> existing Level1 profiler, five Graph replays -> full/exact
oracle ->10 warmup +5x100 unprofiled -> full/exact oracle. Source, identity,
occupancy, stream and120s/process/2GiB allocated budget remain enforced.
Three-process total local trace budget1GiB; no retries/B1/TP4 or new kernel.

Any Event OR Wall abs(after median-before median)/before median>=2% flags
Profiler interference; it is not removed or repaired by selecting trials.
Correctness/identity/source/Graph/resource/conversion failure stops later runs.
Control drift alone is recorded and the fixed B0 panel continues.

Device CSV timestamps use Decimal subtraction before conversion to relative
microseconds; epoch-scale float subtraction would lose small handoff gaps.
Kernel CSV joins exact task-start/stream/task IDs. Full task order and actual
native symbols/task/model/stream IDs are retained, but device/host identifiers
and absolute timestamps stay local. Task/model IDs are trace-local identifiers,
not a claim that an unavailable binary kernel-entry ID was measured.

Replay association uses five MODEL_EXECUTE starts and exact one-GEMM/one-consumer
task groups between successive model starts. Graph span is the **observed device
envelope** from MODEL_EXECUTE start to last non-profiling task stop in that group,
not the MODEL_EXECUTE task's own duration or host API time. Notify/control tasks
may overlap compute and are not added as serial overhead. Missing ACL->NPU flow
links limit causal association; profiled launch gaps cannot be substituted for
unprofiled100-replay Event latency. Profile/control perturbation may leave
unprofiled critical-path attribution UNKNOWN even when timestamps are readable.

## Kernel durations: not invariant

All values are microseconds. Compute/control medians use five profiled replays;
Event/Wall medians use five unprofiled100-replay trials per phase.

| Process | Event before | Event after | GEMM | Consumer | Handoff gap | Graph envelope |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 45.578198 | 45.648398 | 23.22 | 18.32 | 0.02 | 45.02 |
| 2 | 52.831998 | 52.766800 | 23.90 | 24.76 | 0.04 | 52.22 |
| 3 | 42.649202 | 42.728000 | 24.12 | 14.80 | 0.02 | 42.34 |

GEMM medians span23.22-24.12us (about3.9% max/min change), not perfectly
constant. Consumer medians span14.80-24.76us (about67.3% max/min change), a much
larger component change. These are descriptive observations, not component-level
statistical significance or a newly invented stability threshold.

All three use the same observed native symbols/dispatch and trace-local IDs:
- GEMM: `MatMulV2_ND_ND_FP16_FP16_false_true_all_98513`, dispatch
  `aclnnMatmul_MatMulCommon_MatMulV2`, stream13/task0/model49, block23.
- Consumer: `split_qkv_rmsnorm_rope_kernel`, stream13/task1, block48;
  model field4294967295 is an exporter sentinel, not a proven real model ID.

Kernel attributes verify BF16/ND; the FP16 substring in the native symbol is
not the input dtype. No standalone conversion/copy task is observed.

## Handoff gap: no macroscopic expansion

All15 measured GEMM-end -> Consumer-start gaps are0.02-0.04us, at the exported
timestamp granularity. They cannot descriptively account for multi-us Event
changes in this panel. This is a device scheduling boundary gap, not proof of
zero producer/consumer memory traffic or zero hidden kernel-internal cost.

| Process | MODEL_EXECUTE task | NOTIFY_WAIT task | NOTIFY_RECORD task | Prefix to GEMM | Completion tail |
|---|---:|---:|---:|---:|---:|
| 1 | 3.02 | 35.56 | 0.02 | 2.92 | 0.50 |
| 2 | 3.04 | 43.56 | 0.02 | 2.96 | 0.48 |
| 3 | 3.04 | 33.58 | 0.02 | 2.94 | 0.48 |

Each trace has27 device tasks: profiling-enable, five repetitions of
MODEL_EXECUTE/GEMM/NOTIFY_WAIT/Consumer/NOTIFY_RECORD in **start-time order**,
then profiling-disable. Caller/control stream47 overlaps compute stream13;
NOTIFY_WAIT ends after Consumer and overlaps compute, so it is not an independent
serial penalty to add to GEMM+Consumer. Prefix/tail do not show a large change.

## Graph span versus Event: descriptive agreement

Reference Event=(before median+after median)/2 is45.613298,52.799399,42.688601us.
Observed device envelopes are45.02,52.22,42.34us; residuals are approximately
0.59,0.58,0.35us. This residual is **not** a calibrated launch overhead estimate:
profiled single-replay envelopes and unprofiled100-replay batches differ.

For process2 versus3, Consumer median increases9.96us, GEMM decreases0.22us,
Graph envelope increases9.88us and reference Event increases10.11us.
**OBSERVED association:** most of this panel's changing envelope is in Consumer
execution, not handoff/prefix/tail. It is not a proof that Consumer intrinsic
instructions, clocks, cache placement or memory latency caused the difference.
The underlying cause and exact unprofiled critical-path attribution are UNKNOWN.

| Process | Event after/before drift % | Wall before | Wall after | Wall drift % | >=2% interference gate |
|---|---:|---:|---:|---:|---|
| 1 | +0.154021 | 47.080100 | 47.197610 | +0.249596 | Not triggered |
| 2 | -0.123406 | 54.188130 | 54.266730 | +0.145050 | Not triggered |
| 3 | +0.184757 | 44.213560 | 44.304860 | +0.206498 | Not triggered |

No persistent before/after baseline disturbance is detected by the unchanged2%
gate. This does **not** prove absence of profiler-only perturbation. Missing
ACL-to-NPU flow events are reported in all three exports, alongside the known
text-export op_attr and RECORD-stop warnings. Actual device tables contain all
five replay groups and10 compute kernels per process, but causal host/device
links remain incomplete. No clock counters or kernel-internal attribution were
collected; these limitations prevent upgrading association to a proven cause.

## Receipts

Three fixed-order processes exit0 within120s/process; full original eager/Graph/
changed-buffer oracle and pre/post-profile exact snapshots pass. Process2/3
QKV/Q/K/V exactly match process1. Input/runtime/source fingerprints and physical
identity match throughout. Peak Torch allocation28,360,704bytes (<2GiB), stable
ND weight pointer/format. Final device healthy/idle, production checkouts clean.
No retry, B1, A/B/A, kernel/environment edit or formal admission.21 CPU tests
pass, including epoch-scale Decimal precision and overlapping-control fixtures;
the original Event/Wall measured AST is unchanged from `39d5b9f`.

Public data: [process1](../results/m0b_timeline_20261017/process1.json),
[process2](../results/m0b_timeline_20261017/process2.json),
[process3](../results/m0b_timeline_20261017/process3.json). Each retains every
unprofiled trial, every profiled replay, full relative device task order and
actual native/task/model/stream identifiers, plus raw-file checksums.
Reducer: `python -S experiments/m0b/timeline_summary.py <local-run-directory>`.
Raw trace/CSV/export logs, original absolute timestamps, device/host identities,
source copies/executed patch/manifest and checksum index remain local in
`m0_receipts/m0b_timeline_20261017/`. The manifest pins that index.
