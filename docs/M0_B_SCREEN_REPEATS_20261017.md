# Bounded repeats of the final M=8 screen

Owner requested additional repetitions after `228d971`. Preregistered two more
rounds, two fresh processes each, fixed order round2_process1, round2_process2,
round3_process1, round3_process2. Original round1 remains two INVALID results.
No replacement, cherry-picking, threshold/harness change or implicit M=1 gate.

Identical M=8 seed0 three-weight/three-Graph B0 ND / shadow ND / B1 NZ design,
five palindromic `A S S A | A B B A | A S S A` cycles,10 warmups+100 replays per
slot, three Event API prewarm pairs, full original oracle,120s/process,2GiB.
Both Event/Wall control thresholds stay2%; gain remains the median of all five
paired Event reductions, not a selected metric/window. Three co-resident weights
are still an experimental fact, not a production memory-neutral configuration.
`formal_approved=false`. No profiler, new kernel, integration, M=1 or TP4.

All four processes will be retained, including INVALID. Source/oracle/identity/
Graph/resource failure stops later runs; control INVALID does not trigger retry.
## All results, including the original round

| Round/process | Control status | Event gain % | Max shadow Event difference abs % | Max B0 bracket drift Event / Wall % |
|---|---|---:|---:|---:|
| 1/1 original | INVALID | RAW 1.339 | 2.992 | 0.732 / 12.874 |
| 1/2 original | INVALID | RAW 7.826 | 1.941 | 0.871 / 2.196 |
| 2/1 added | INVALID | RAW -2.400 | 0.749 | 38.710 / 37.541 |
| 2/2 added | VALID screen | 2.599 | 1.778 | 0.695 / 1.492 |
| 3/1 added | VALID screen | 12.425 | 1.630 | 0.591 / 1.168 |
| 3/2 added | VALID screen | 12.206 | 0.820 | 0.624 / 1.221 |

INVALID rows have null accepted gain. Their RAW arithmetic is retained, not
treated as performance evidence. All windows and all rounds remain present.
Added round2 failed the per-panel gate; **added round3 passed it**: both fresh
processes valid, each chain Event reduction>=5%. This was the fixed last round,
not selected windows or a replacement for previous failures. Its paired Wall
reductions also remain available in the full records.

### Interpretation

The earlier blanket conclusion "no passing panel" is no longer accurate: one
complete, preregistered repeat panel now passes the original quick gate. But
across all three rounds,3/6 processes are INVALID, and the other valid process
improves only2.599%. This is **mixed reproducibility**, not an established stable
12% improvement, statistical significance, formal GO or service benefit.
No averaging away failed controls, relaxing thresholds or mechanism falsification.
The source of inter-process variation remains UNKNOWN; no extra diagnosis ran.

The next bounded step under the original screening sequence would be M=1,
preserving equivalent controls and all prior failures. It was **not run in this
repeat-only scope**; neither integration nor fusion development was started.
MLP Materialization remains the alternative if follow-up does not establish a
usable reproducible benefit. Current data justify reviewing the pause, not
erasing the earlier NO-GO checkpoint or silently declaring formal admission.

### Evidence

Four added processes exit0, all12 added complete Graphs pass full original
oracles before/after and paired checks.23 CPU tests pass. Same physical card,
source hashes/runtime/input fingerprints as original round1; harness and all
thresholds unchanged. Three weights remain co-resident; allocated83,258,880bytes
and peak110,184,960bytes in each added process (<2GiB). No timeout or safety
failure; final device idle. No profiler, extra seed, production change or M1.

[All-round summary](../results/m0b_screen_repeats_20261017/all_rounds.json) lists
every process, gate and accepted/RAW gain. The same directory contains four
full sanitized process records with all240 added Event/Wall rows and numeric
oracles; original120 rows stay in `results/m0b_final_screen_20261017/` unchanged.
Raw commands/console/progress/snapshots/exit stamps/manifest/checksum index stay
local in `m0_receipts/m0b_screen_repeats_20261017/`. Source is the unmodified
screen harness at `228d971`; reducer remains `paired_screen.assess(trials)`.
