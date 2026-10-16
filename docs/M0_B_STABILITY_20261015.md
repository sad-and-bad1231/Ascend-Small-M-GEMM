# M0-B B0-only measurement stability: 2026-10-15

Base `14961fd3e04396d90fdf696b3c063f5a2fd6d2ab`. This diagnostic does not
replace or retry the two INVALID prior triplets. `formal_approved=false`.
Preparation began2026-10-15; resumed2026-10-16 (Asia/Shanghai). Filename retains
the requested experiment label. **BLOCKED: physical allocation changed on resume;
zero completed measurement windows.** No B0 stability PASS or A/B/A readiness.

## Static timing review (before hardware)

- Original Event objects are constructed before Wall starts, but native event
  creation/record is lazy. Wall starts before `a.record()`, includes100 replay
  calls, `b.record()`, device-wide synchronization and `a.elapsed_time(b)` query.
  Therefore first-trial Wall overhead is not necessarily device computation.
- Event brackets the current stream. Installed `Event.record(None)` resolves
  `current_stream()`. Installed graph context chooses its internal side stream
  for capture, synchronizes at entry and restores caller stream at exit.
  Replay Python delegates to native; installed native disassembly calls
  `getCurrentNPUStream`. This is consistent with caller-stream replay, not proof
  of all internal model task dependencies. We record capture/caller stream handles
  locally and check caller stream unchanged in every trial. No explicit stream
  switch is added around timing. Device-wide synchronize waits for all work.
- Before timing: eager oracle, three eager capture warmups, capture and first
  replay, graph oracle, five synchronized exact replays, changed-input/position
  oracle, then buffer restoration/replay. Each measurement window retains the
  original10 extra warmups + synchronize, then five100-replay batches. Warmups
  do not record/query timing events, so cannot warm lazy timing-event paths.
- Fresh processes repeat import/context setup, synthetic tensor generation,
  compilation/cache lookup, capture, oracle and teardown. Graph/tensor addresses
  and process scheduling may differ even with matched input/source/runtime.
  Compile/setup is outside timers; persistent layout/cache effects are unknown.

## Preregistered scope

One allocated physical910B2, M=1, seed0, BF16, B0 ND-only, same current consumer.
First one process with three sequential windows on **one capture**, then three
fresh processes with one window each. Four total processes, no retries or B1.
Each process120s maximum and2GiB Torch allocated limit, full original oracle
before windows and full/exact oracle after each window. No profiler.

State snapshots run before warmup and after the fifth trial, never inside the
timed window: read-only npu-smi identity/occupancy/temperature/power/utilization,
DCMI frequency getters, host affinity/load/CPU pressure/cgroup throttling and
process CPU/context switches. These snapshots are not in-window telemetry and
introduce gaps between same-process windows; no claim of uninterrupted workload.

Added host timers split start-event record,100-call replay enqueue, end-event
record, device synchronize and Event query. They are batch spans divided by100,
not per-call distributions. Extra clock reads change instrumentation slightly;
original Event/Wall definitions and10/5x100 counts stay intact. Host enqueue can
overlap device work or include runtime backpressure, so do not add enqueue+Event
or treat synchronization remainder as device compute.

Report both endpoint drift `abs(last-first)/first*100` and spread
`(max(medians)-min(medians))/first_median*100` for Event/Wall. Also retain all
pairwise drifts using the earlier median as denominator; any pair >=2% blocks
new A/B/A readiness. Historical2% threshold is unchanged. No A/B/A this round,
regardless of outcome. Correctness/source/Graph/resource/identity/occupancy
failure stops later hardware. Boundary monitoring cannot exclude brief unseen
interference or clock transitions.

## SAME_PROCESS_VARIATION

**NOT_MEASURED.** The intended one-Graph/three-window control did not reach a
measurement window. One process was started in error after a failed external
identity recheck, then explicitly terminated: exit143. Its only progress entry
is the permission checkpoint, with no oracle, trials or result JSON. No statement
that framework import/device initialization never began is justified.

## CROSS_PROCESS_VARIATION

**NOT_RUN.** All three fresh B0 controls were withheld. No endpoint drift,
spread, enqueue time or performance gain is fabricated. Original M1/M8 timing
pilot receipts and INVALID classifications remain unchanged, with no reruns.

## DEVICE_STATE

Initial and resumed read-only board/mapping queries identify different physical
devices, not just a logical-index rename. Exact identifiers stay local. Both
are910B2; original and current snapshots show healthy/idle devices. Current
allocation has no remaining NPU process after the owned abort.

Initial outside-window getters returned AICore current800/max1800 and HBM1600
(driver API values); vector-frequency query returned-8255/unavailable. Initial
SoC/HBM temperature was39C and power98.0W. These are **idle preparation readings,
not measurement-window conditions**. New-allocation raw device and Host CPU
snapshots are kept separately and cannot explain previous-card timing drift.
No within-window thermal/clock or Host pacing observations exist.

## HOST_ENQUEUE_VS_DEVICE

**UNKNOWN / NOT_MEASURED.** The added enqueue/synchronize/record/query timers
are CPU-tested code only, not hardware-validated metrics. Static scope review
above shows why neither original Event nor Wall is isolated compute time.
First-Wall overhead remains a plausible lazy-event/query effect, not a proven
cause of the original cross-process Event drift.

## ROOT_CAUSE

**Original timing instability: UNKNOWN.** No clock, pacing, compilation/Graph
or competing-resource hypothesis was verified. Do not use allocation change
on resume to retrospectively explain the earlier same-allocation drift.

**Current execution blocker: OBSERVED physical identity mismatch.** The resume
snapshot requested the previous physical card and failed; its retained stdout
file is empty (not valid JSON). The external command sequence did not fail-stop:
`set -o pipefail` does not stop subsequent sequential commands. A failed JSON
identity check was followed by a successful hash check and an erroneously
issued launch. This orchestration error is preserved, not counted as a valid
hardware admission or replaced with a favorable run.

Minimal repair: save even failed-query snapshots, and validate physical identity,
logical mapping, health and occupancy **inside the probe before framework
imports**, as well as before/after each future window. A subprocess regression
test proves identity rejection occurs before any torch/vLLM import. No production
environment, kernel or historical drift threshold was changed.

## A/B/A readiness

**NO: conditions not established.** No performance A/B should run. Continuing on
the new card needs explicit scope amendment and an independent receipt batch,
not relabeling it as the old physical card. If authorized, first perform exactly
the proposed B0-only controls; any Event/Wall pair drift>=2% keeps A/B blocked.
Only if needed and separately authorized, a short B0-only trace (e.g. five
replays in a drifting allocation) could inspect replay-stream ordering and
enqueue gaps. It would not replace unprofiled timings or prove device frequency
history; no profiler has run this round.

## Evidence

18 CPU unit/mock tests and static compilation pass after the guard repair;
17 CPU tests passed before it. Production vLLM/plugin checkouts remain clean.
Guard-fixed code has **not** passed hardware correctness or resource tests;
120s/2GiB remain required limits, not reported measured successes.

Local directory: `m0_receipts/m0b_stability_20261015/`, including original source
copies/dirty patch, pre/post-fix hashes, initial/new-allocation state, aborted
console/progress/exit143 and incident chronology. No failed evidence is deleted.
[Sanitized status receipt](../results/m0b_stability_20261015/status.json) contains
only de-identified counts/status/checksums. Local checksum index is pinned in
the manifest; no raw device/host logs or tensor snapshots are published.
