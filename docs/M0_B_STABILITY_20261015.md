# M0-B B0-only measurement stability: 2026-10-15

## New independent batch authorized2026-10-16

After the blocked checkpoint below, the owner explicitly authorized the newly
allocated single910B2 for an independent B0-only batch, based on `bb3c82c`.
Same four-process/M1/seed0/10-warmup/5x100 scope,120s/process,2GiB, no B1 or
Profiler. New local receipts: `m0_receipts/m0b_stability_new_allocation_20261016/`.
Physical identity is frozen at this batch's initial snapshot and checked before
framework import and outside every window. The older aborted batch stays intact;
no cross-card timing comparison or reclassification of prior INVALID data.
**Completed: B0 cross-process instability reproduced; A/B/A remains BLOCKED.**
Four processes exit0, six windows/30 trials, no retries or profiler. Full original
eager/Graph/changed-buffer oracle and all six post-window full/exact checks pass;
the three fresh-process QKV/Q/K/V snapshots exactly match the same-process B0.
Input/source/runtime fingerprints match across all four. Executed probe is the
unchanged guard-fixed `eb9e0887b30a9cad504b610c61a450b21700ba9553542bbb01ec427343648c68`.
All processes used the120s timeout; observed maximum allocated Torch memory was
28,360,704bytes, below2GiB. Weight pointer/ND format stayed stable in every window.
No production source/package/environment change.19 CPU tests pass. This small
diagnostic panel establishes no formal statistical significance or M1/M2 GO.

### SAME_PROCESS_VARIATION

One capture, three sequential windows. Units below are microseconds per replay;
each trial is a100-replay batch, all five trials retained in each median.

| Window | Event median | Wall median | Host enqueue median | Synchronize median |
|---|---:|---:|---:|---:|
| 1 | 40.360999 | 41.929060 | 11.328830 | 29.060300 |
| 2 | 40.551000 | 42.011370 | 11.510040 | 29.429010 |
| 3 | 40.503602 | 41.744370 | 11.196130 | 29.255000 |

Event spread/first=0.470753%, endpoint drift=0.353319%.
Wall spread/first=0.636790%, endpoint drift=0.440482%.
All pairwise median drifts stay below2%; this is only this Graph/allocation.
Snapshot/oracle gaps separate the windows; no uninterrupted-load claim.

### CROSS_PROCESS_VARIATION

Three fresh processes after the same-process control, same one-window10/5x100
protocol, each referencing the first B0 snapshot for exact-output comparison.

| Process | Event median | Wall median | Host enqueue median | Synchronize median |
|---|---:|---:|---:|---:|
| 1 | 42.265801 | 43.858510 | 11.343840 | 30.800140 |
| 2 | 48.508401 | 50.003940 | 11.676250 | 36.798960 |
| 3 | 46.497202 | 47.998090 | 10.982830 | 35.486340 |

Event spread/first=**14.769859%**, endpoint drift=**10.011405%**.
Wall spread/first=**14.011944%**, endpoint drift=**9.438488%**.
All three pairwise Event and Wall drifts exceed2%, including process2->3
(Event4.146084%, Wall4.011384%). No endpoint selection rescues this panel.
Within-window Event range/median is below1.20% in all six windows: tight batches
do not imply reproducible process baselines. No optimization gain is measured.

### DEVICE_STATE

All pre-import and window-boundary checks match **one newly authorized physical
device**, with logical0 mapping and no unexpected NPU processes. Device is
healthy/idle after completion. Window-boundary SoC/NDIE40-41C, HBM39-40C;
power102.8-103.7W. DCMI always reports AICore current800/max1800 and HBM1600
(driver values); vector getter remains unsupported, rc=-8255.

These queries are outside windows and follow multi-query snapshot latency:
**they do not establish constant in-window clock, power or temperature**.
No monitoring command ran inside a timed batch. Host affinity192 CPUs,
cgroup quota22 CPU-equivalents; nr_throttled/throttled_usec remain0. Boundary
Host load1m14.53-22.68 and CPU pressure are retained. Load is host-wide, not
this process's utilization; host pressure/context-switch snapshots include
oracle/monitoring gaps and cannot exclude short in-window scheduling stalls.

### HOST_ENQUEUE_VS_DEVICE

Across fresh processes Host enqueue medians stay10.983-11.676us while Event
changes42.266-48.508us and synchronization remainder30.800-36.799us.
**OBSERVED:** most cross-process change appears in the device-completion wait,
not a commensurate rise in the100-call Host enqueue span. This weighs against
simple average Host enqueue throughput as the sole cause, but enqueue may overlap
work or hide asynchronous runtime queues; neither Event nor sync is pure compute.

First-trial Wall overhead is separately localized. Compare trial1 vs trial2
in the first window of each process, without removing either trial:

| Process | Wall excess us/replay | start-record excess | Event-query excess |
|---|---:|---:|---:|
| same-process | 11.559250 | 7.570160 | 2.051950 |
| fresh1 | 9.623300 | 6.074230 | 1.960940 |
| fresh2 | 10.886930 | 7.202960 | 2.069250 |
| fresh3 | 10.874730 | 6.664350 | 2.674250 |

The two Event API paths account arithmetically for about83-86% of first-Wall
excess. Their lazy initialization/cache behavior is **INFERRED**, not profiled.
These entries are **single-call/batch overhead divided by100**, not the actual
single API call latency. They explain a Wall-specific cold cost, not why the
Event baseline shifts across processes. Event objects are allocated each trial;
only the first Event usage/query in a process is markedly cold in these receipts.
Original warmups replay Graph but do not warm timing-event record/query paths.

### ROOT_CAUSE

**UNKNOWN for cross-process Event drift.** Observed facts narrow the problem to
an allocation/process-sensitive component-chain span with reproducible outputs;
they do not prove device clocks, kernel dispatch, Graph model scheduling,
memory/cache placement or competing-resource causality. Cold Event API overhead
is a measured contributor to first Wall, not a demonstrated explanation of
Event drift. Runtime source identity is unchanged; fresh compile/capture/cache
lookup is outside timers, but persistent effects remain possible.

### A/B/A readiness and next gate

**NO.** Pure B0 fails the unchanged2% gate. No B1/A/B/A, extra seed, retry,
warmup tuning, profiler or formal approval was introduced. Old-card pilot remains
INVALID and cannot be compared directly to this new card's absolute latency.
If separately authorized, the smallest next check is B0-only short device traces
(e.g.5 replays/allocation) to compare actual GEMM/consumer dispatch, individual
kernel durations, model execution span and launch gaps between fresh processes.
Use them only for mechanism attribution, not performance claims; no fixed-clock
or governor changes are proposed. This round stops here.

### Data and reproducibility

[Sanitized full-precision summary](../results/m0b_stability_new_allocation_20261016/summary.json)
retains all30 trials, all seven timing fields, boundary state aggregates and
pairwise calculations. [Reducer](../experiments/m0b/stability_summary.py) verifies
five trials/window, identity/provenance, full-oracle receipts and the exact
Wall decomposition. Command: `python -S experiments/m0b/stability_summary.py
m0_receipts/m0b_stability_new_allocation_20261016` (on one line).
Raw tensors, process/device identities, all queries/console/progress/exit stamps,
source copies, executed manifest/patch and checksum index remain local.
The earlier aborted-batch and pilot checksum indexes still verify unchanged.

## Earlier blocked checkpoint (retained history)

Base `14961fd3e04396d90fdf696b3c063f5a2fd6d2ab`. This diagnostic does not
replace or retry the two INVALID prior triplets. `formal_approved=false`.
Preparation began2026-10-15; resumed2026-10-16 (Asia/Shanghai). Filename retains
the requested experiment label. **BLOCKED: physical allocation changed on resume;
zero completed measurement windows.** No B0 stability PASS or A/B/A readiness.

### Static timing review (before hardware)

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

### Preregistered scope

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

### Historical SAME_PROCESS_VARIATION

**NOT_MEASURED.** The intended one-Graph/three-window control did not reach a
measurement window. One process was started in error after a failed external
identity recheck, then explicitly terminated: exit143. Its only progress entry
is the permission checkpoint, with no oracle, trials or result JSON. No statement
that framework import/device initialization never began is justified.

### Historical CROSS_PROCESS_VARIATION

**NOT_RUN.** All three fresh B0 controls were withheld. No endpoint drift,
spread, enqueue time or performance gain is fabricated. Original M1/M8 timing
pilot receipts and INVALID classifications remain unchanged, with no reruns.

### Historical DEVICE_STATE

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

### Historical HOST_ENQUEUE_VS_DEVICE

**UNKNOWN / NOT_MEASURED.** The added enqueue/synchronize/record/query timers
are CPU-tested code only, not hardware-validated metrics. Static scope review
above shows why neither original Event nor Wall is isolated compute time.
First-Wall overhead remains a plausible lazy-event/query effect, not a proven
cause of the original cross-process Event drift.

### Historical ROOT_CAUSE

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

### Historical A/B/A readiness

**NO: conditions not established.** No performance A/B should run. Continuing on
the new card needs explicit scope amendment and an independent receipt batch,
not relabeling it as the old physical card. If authorized, first perform exactly
the proposed B0-only controls; any Event/Wall pair drift>=2% keeps A/B blocked.
Only if needed and separately authorized, a short B0-only trace (e.g. five
replays in a drifting allocation) could inspect replay-stream ordering and
enqueue gaps. It would not replace unprofiled timings or prove device frequency
history; no profiler has run this round.

### Historical evidence

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
