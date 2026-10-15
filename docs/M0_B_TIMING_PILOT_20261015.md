# M0-B bounded timing pilot: 2026-10-15

**Both triplets INVALID: no drift-screened component-chain gain established.**
This is owner-authorized `DIAGNOSTIC_ONLY`, not Issue #20 admission:
`formal_approved=false`. No formal significance or M1/M2 GO is claimed.
Prior [distinct ND/NZ dispatch evidence](M0_B_DISPATCH_20261015.md) remains valid;
invalid timing is not evidence that the NZ mechanism itself fails.

## Frozen method

Base `71f310a1769676d4707e53b5c5196f539a3bfa9c`. Same current 910B2/runtime
and existing consumer; BF16 X[M,5120], W[2560,5120], seed0, no bias.
B0 uses official ND GEMM; B1 uses only this QKV weight in NZ, not a global
weight policy. Each M has exactly one B0-before / B1 / B0-after triplet,
three fresh processes, matched input/source/runtime fingerprints within it.
M=1 was reviewed and marked INVALID before proceeding to authorized M=8.
No retries, dropped trials, extra seeds/shapes, profiler or production edits.

Each process passed the complete existing oracle before timing, replayed
10 warmups, then 5 trials x 100 Graph replays, then passed the complete oracle
again. B1 and rollback QKV/Q/K/V exactly match their first B0 snapshots.
Consumer geometry is 16 Q heads, 2 KV heads, head dimension128.
M-specific random tensors need not match across different M.

All numbers below are microseconds **per replay**, dividing each batch by100.
NPU Event measures the replay stream span, not isolated GEMM+consumer kernel
duration; enqueue starvation/idle can contribute. Wall includes host enqueue,
event bookkeeping/query and synchronization. Receipt checkpoints and final
oracle run outside the measured interval. No kernel-time subtraction is made.

Per metric: reference=(median B0-before + median B0-after)/2;
raw reduction=(reference - median B1)/reference x100;
drift=abs(median B0-after - median B0-before)/median B0-before x100.
The preregistered conservative gate invalidates the whole triplet if **either
Event or Wall drift >=2%**. Accepted gain is null for both metrics when invalid.
These five batches share one process, not five independent allocations.

## Medians and drift

| M | Arm | Event us | Wall us |
|---|---|---:|---:|
| 1 | B0-before | 40.211802 | 41.863730 |
| 1 | B1 | 39.447000 | 40.975720 |
| 1 | B0-after | 53.244600 | 54.687770 |
| 8 | B0-before | 48.731799 | 50.268590 |
| 8 | B1 | 38.611200 | 40.249710 |
| 8 | B0-after | 51.088200 | 52.476830 |

| M | Metric | B0 reference us | Rollback drift % | RAW reduction % (INVALID) | Accepted gain |
|---|---|---:|---:|---:|---|
| 1 | Event | 46.728201 | 32.410383 | 15.582027 | NONE |
| 1 | Wall | 48.275750 | 30.632817 | 15.121526 | NONE |
| 8 | Event | 49.909999 | 4.835447 | 22.638348 | NONE |
| 8 | Wall | 51.372710 | 4.392882 | 21.651573 | NONE |

The raw reductions are retained arithmetic, **not usable speedup estimates**.
Both fail even the primary Event-only gate. Full-precision sanitized receipts:
[M=1](../results/m0b_timing_pilot_20261015/m1.json),
[M=8](../results/m0b_timing_pilot_20261015/m8.json).

## All trials

| M | Arm | Trial | Event us | Wall us |
|---|---|---|---:|---:|
|1|B0_before|1|40.215402|50.582580|
|1|B0_before|2|40.217199|42.016230|
|1|B0_before|3|40.188398|41.733630|
|1|B0_before|4|40.196600|41.863730|
|1|B0_before|5|40.211802|41.673330|
|1|B1|1|39.458799|51.235500|
|1|B1|2|39.422600|40.994820|
|1|B1|3|39.497600|40.974120|
|1|B1|4|39.220400|40.975720|
|1|B1|5|39.447000|40.884920|
|1|B0_after|1|53.197598|64.810440|
|1|B0_after|2|53.250599|54.775770|
|1|B0_after|3|53.244600|54.687770|
|1|B0_after|4|53.265600|54.613970|
|1|B0_after|5|53.238001|54.590160|
|8|B0_before|1|48.687601|58.500630|
|8|B0_before|2|48.719001|50.450490|
|8|B0_before|3|48.731799|50.209990|
|8|B0_before|4|48.763399|50.268590|
|8|B0_before|5|48.783798|50.150190|
|8|B1|1|38.611200|49.121580|
|8|B1|2|38.606400|40.378010|
|8|B1|3|38.442199|40.072620|
|8|B1|4|38.669200|40.219220|
|8|B1|5|38.636999|40.249710|
|8|B0_after|1|51.202598|62.641520|
|8|B0_after|2|51.276398|53.028250|
|8|B0_after|3|51.073198|52.476830|
|8|B0_after|4|51.088200|52.449440|
|8|B0_after|5|51.067400|52.426740|

## Interpretation

- **OBSERVED component-chain gain:** none established after the drift gate.
  B1's measured medians are lower, but drifting B0 endpoints invalidate attribution.
- **MEASUREMENT NOISE / instability:** Event within-arm range/median is below
  0.71% in all six processes, yet cross-process rollback drift reaches32.41%
  and4.84%. Tight repeated batches do not ensure stable baseline allocations.
  Every first Wall trial is elevated; all are retained. Its cause, and the cause
  of baseline drift, remain unmeasured, not assigned to clocks or interference.
- **POSSIBLE consumer cost:** same consumer code does not prove invariant cache
  or scheduling interactions. This chain-only pilot cannot measure a consumer
  penalty, benefit or their share of total time.
- **UNKNOWN critical-path attribution:** Event/Wall spans cannot identify
  GEMM, consumer, launch gaps or hidden internal costs individually. Earlier
  profiled durations must not be subtracted from this unprofiled pilot.

The next useful gate is a separately scoped baseline-stability diagnostic before
any new comparative performance claim, not automatic seed expansion or M1/M2.
There is no additional hardware authorization implied by this recommendation.

## Receipts and checks

Six processes exit0 within their120s timeout. Weight pointers/formats remain
stable (2/29/2); maximum observed load allocation54,608,384bytes and timing
allocation28,564,992bytes are below2GiB. This is Torch allocated memory, not a
claim about all CANN private caches. Device was idle before/between/after;
production source checkouts remain clean. No correctness/Graph/source/resource
failure occurred; both measurement-invalid results are preserved without retry.

15 CPU unit/mock tests pass; static compilation and sanitized/raw receipt
cross-checks pass. The harness minimally adds per-trial progress logging,
post-timing oracle/resource checks and Wall median reporting outside timing.
[Summary helper](../experiments/m0b/timing_summary.py) rejects missing trials,
scope/source mismatches and profiler mixing; tests cover the2% boundary.

Raw console/progress/output snapshots, exit stamps, source copies/dirty patch,
M1-before-M8 review and checksum index stay local under
`m0_receipts/m0b_timing_pilot_20261015/`. Public JSON includes raw-result hashes,
all30 trial values and only sanitized summaries. Executed probe SHA256:
`7d1e3d4097f3ee8befc97b1a509da6614efdec3f0d69f219b20cbc880c8a2512`.
The manifest pins the final local receipt index; no original device logs or
snapshots are published. Approval remains false.
