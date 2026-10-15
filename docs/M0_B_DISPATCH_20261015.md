# M0-B Dispatch and Conversion Diagnostic

**DIAGNOSTIC_ONLY; formal_approved=false.** Base: `5326ca9`, same personal branch.
No unprofiled timing, 10-seed matrix, service capture, kernel edits or upgrades.
Current software identity was checked by the existing harness, not re-audited.

## Actual Dispatch: OBSERVED

Each row is one fresh process, seed0, BF16, K=5120/N=2560, five profiled replays.
Full labels, input metadata, counts and raw hashes: [sanitized receipts](../results/m0b_dispatch_20261015/).

| M | Arm | CANN dispatch | Native GEMM suffix | Inputs -> output | GEMM / consumer launches | Pair oracle |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | B0 | aclnnMatmul | ND_ND...98513 | ND;ND -> ND | 5 / 5 | PASS |
| 1 | B1 | aclnnMatmulWeightNz | NDNZ_ND...229585 | ND;FRACTAL_NZ -> ND | 5 / 5 | exact B0 match |
| 8 | B0 | aclnnMatmul | ND_ND...98513 | ND;ND -> ND | 5 / 5 | PASS |
| 8 | B1 | aclnnMatmulWeightNz | NDNZ_ND...229585 | ND;FRACTAL_NZ -> ND | 5 / 5 | exact B0 match |

Exact device task symbols and CANN names (not guessed from weight metadata):

```text
B0: aclnnMatmul_MatMulCommon_MatMulV2
    MatMulV2_ND_ND_FP16_FP16_false_true_all_98513
B1: aclnnMatmulWeightNz_MatMulCommon_MatMulV2
    MatMulV2_NDNZ_ND_FP16_FP16_false_true_all_229585
Consumer, both arms: split_qkv_rmsnorm_rope_kernel
```

Both GEMMs use 23 blocks; the existing AI_VECTOR_CORE consumer uses 48 blocks.
The native name contains `FP16`, but tensor dtype AND CANN input metadata are
**BF16 / DT_BF16**. Do not reinterpret a symbol's name as the measured dtype.
The NZ input physical shape is `[320,160,16,16]`; logical W remains `[2560,5120]`.

Full oracle retained: FP32 GEMM/reference checks, consumer/chain reference,
finite/shape/dtype/ND contiguity, exact V, exact eager/graph and five fixed-buffer
replays, changed X/positions, paired outputs. Trace completion adds another full
oracle and exact snapshot comparison **outside** the profiled window. All passed.
QKV relative L2 vs FP32: M1 **0.00166295**, M8 **0.00164346**; maximum consumer
absolute error across original/changed inputs: **0.0237813 / 0.0262570**.

## Hot Conversion and Residence

- **OBSERVED:** zero standalone TransData/cast/transpose/memcpy candidates in
  device tasks or duration-type runtime events in all four replay windows. Exactly
  two compute tasks per replay: GEMM then the unchanged consumer. This conclusion
  uses the complete task inventory as well as name matching, not a kernel-name
  heuristic alone.
- **OBSERVED:** B1's explicit weight `npu_format_cast(...,29)` is load-time only,
  outside the replay trace. Weight/output formats are 29/2, respectively. No
  explicit activation repair is inserted. The load cast's device kernel/time was
  NOT profiled; its existence is established by the call and physical-format check.
- **OBSERVED:** weight pointer and format are unchanged before/after each trace;
  allocated before/after/peak during trace is identical: M1 **28,349,952 bytes**,
  M8 **28,564,992 bytes**, for both arms. After releasing the ND weight, B0/B1
  allocated bytes match: M1 **27,273,728**, M8 **27,345,408**. NZ load transient
  peaks are **54,536,704 / 54,608,384 bytes**, not zero-copy allocation claims.
- **INFERRED:** these observations support one explicitly resident QKV weight
  with no repeated external format-conversion task in the tested replay window.
- **UNKNOWN:** GEMM-internal packing/layout work, private CANN caches outside the
  Torch allocator, memory traffic and pipeline waits. No disassembly/AIC counter
  study was done. Do not generalize to zero internal conversion cost or speedup.

## Graph Events and Export Limits

Each arm has five `M0B_GRAPH_REPLAY_*` CPU scopes, five actual
`AscendCL@aclmdlRIExecuteAsync` events, five `MODEL_EXECUTE` device tasks, five
GEMMs and five consumers. Also five NOTIFY_WAIT and five NOTIFY_RECORD control
tasks, plus one profiling-enable/disable pair. Control tasks are not additional
compute kernel launches. Ten `HostToDevice` entries have `ph=f`: these are flow
end markers, **not ten host-to-device memcpy operations**.

All exports warn that `op_attr` is ignored for text output and that stopping from
RECORD may be incomplete; the ACL-to-NPU flow parser also reports missing links.
Preserved as limitations, not suppressed or called a clean profiler PASS. The five
API/model/compute counts independently reconcile, and device labels/formats are
present in both kernel/task CSVs. **UNKNOWN:** complete CPU-to-device flow-arrow
mapping. No extra hardware rerun or parser/framework patch was made to hide this.

## Fallback and Custody

M16/no-bias and M1/GEMM-bias each ran B0/B1 correctness-only, both physical weight
format2, complete oracle PASS and exact QKV/Q/K/V pair equality. No NZ trial or
profile was silently substituted. [Fallback receipts](../results/m0b_dispatch_20261015/fallbacks.json).

The session was interrupted after initial fallback work. On resume the allocated
physical device changed (still one idle 910B2); earlier receipts are retained but
**excluded from the final matched panel**. All final eight invocations ran on the
current allocation with saved exit0, <=120s/process. No cross-card A/B pairing.
Final device check is idle; golden vLLM/Ascend sources remain clean. Raw artifacts
are under `m0_receipts/m0b_dispatch_20261015/`; only sanitized JSON is published.
The local SHA256 index covers 546 files; its hash is pinned in the manifest.
Total local evidence is under 5MiB versus a 1GiB trace cap; maximum observed load
allocation is under 55MB versus a 2GiB allocator cap.

Executed probe SHA256: `0132ddf60665b237c174883bcf42206c6ed10f59921cbc4c4d94fd3e2b4ec4e6`
for profiles, `01ad3d7e35345e5edcc7d765865d491f1754812f8cf371f1b19b00aeb5686969`
for fallbacks. Tracked dirty patches and source/runtime fingerprints are local;
the subsequent analyzer adds no device work. A CPU inspection initially assumed
a dict-shaped trace; actual export is an array. The small stdlib analyzer now
accepts both shapes; no device experiment was failed or repeated for that reader
error. All original profiler warnings/logs remain local.
Final CPU verification: py_compile exit0 and **12 unit/mock tests PASS**.

## Timing Readiness: Proposal Only

**READY for a bounded, separately authorized pilot, not FORMAL admission.**
Actual distinct NZ dispatch and equivalence are now evidenced for M1/M8. No
profiler durations are used as performance gains, and no gain is asserted.

Minimal next pilot: seed0, M1 and M8, **one fresh-process B0/B1/B0 triplet per M**,
same inputs/current allocation, no profiler, full oracle; 10 warmup replays and
5 trials x100 replays/arm. Record every event/wall trial, allocation, paired
ratios and rollback drift; >=2% baseline drift invalidates the triplet. One
triplet is descriptive only, not an independent-repeat confidence claim.
Retain the same 120s/process, allocator budget and fail-stop checks. No 10-seed
expansion. This proposal is **NOT executed or added to the current whitelist**;
an owner-authorized diagnostic pilot and an advisor-approved formal experiment
remain distinct permissions.
