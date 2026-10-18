# Formal M0 admission readiness

Status: **M0 INCOMPLETE / review and TP4 blocked; formal_approved=false**.
Base personal commit `7b79971`; authority is intellistream main
`cb095ea9dc335507ffdf7aedfb5d761fe6b5d88c`, `RESEARCH_CONTRACT.md`,
2026-10-14 handoff and Issue #20. Saved authenticated Issue #20 read has zero
comments; no advisor approval is recorded. Draft has not been posted.

## Completed evidence

| Evidence | Observed boundary |
|---|---|
| M0-A | 12 small inputs hash/size verified; six request windows reaggregated; 13 attribution and two custody CPU cases plus capture self-test pass |
| M0-B | Synthetic per-rank BF16 full oracle, real ND/NZ dispatch and Graph replay; M=16/bias correctness fallback, not TP4 or checkpoint output validation |
| Initial timing | M=1/M=8 rollback drift 32.41%/4.84%, both INVALID; retained |
| Stability/timeline | B0 same-Graph variation 0.47%, cross-process 14.77%; profiled consumer variation observed, underlying cause UNKNOWN |
| Six M=8 seed0 screens | 3 INVALID; 1 VALID +2.5989%; 2 VALID +12.4252%/+12.2058%; all raw trials and failures retained |

The screen's three full weight allocations temporarily co-resided. It establishes
neither service memory neutrality nor a stable population effect. B1 is a
historical control; **B2 fused QKV-consumer is the research mechanism, not yet
implemented**. Personal NO-GO / switching to MLP is superseded, not an upstream
research decision. No further seed0 screening or other topic is planned.

## Historical reproduction gaps

PR #14 `11aca1de03129d6f4f56d846864af20ab35eaee8` and all three pinned fork
commit objects are available locally. Sparse PR checkout has no initialized
third_party runtime. Exact historical container/environment is not recovered.
Pinned source objects are not evidence of an installed, ABI-compatible runtime.
The old NZ patch applies to pinned files, not the current installed plugin.

Missing at historical execution: executed dirty-runner patch/hash, container
digest, immutable checkpoint revision/content hash, full generated token IDs.
These cannot be reconstructed from PR HEAD or completion counts. They remain
UNKNOWN unless independent historical receipts are recovered; a new run cannot
retroactively repair them. Sixteen indexed CSVs (597,659,552 bytes) have not been
downloaded, hashed or reaggregated; rollback CSVs lack complete retained custody.

Historical matched **profiled** TPOT medians 31.893/31.478/31.938 ms imply
approximately 1.30% B1 improvement, below 2%; consumer +2.33585% is derived from
the retained manifest, not freshly verified CSV. The stale +16-26% narrative in
that manifest is not this run's measured consumer penalty.

## Model and device readiness

Found local Qwen/Qwen3-32B. CPU inspection confirms 17 shards, 707 indexed
BF16 tensors, complete nonoverlapping header offsets and expected file sizes;
payload 65,524,246,528 bytes. Config: K=5120, 64 Q heads, 8 KV heads, dimension
128, 64 layers; TP4 gives Q=16/KV=2, N=2560 per rank. Config/index/tokenizer
hashes are in the manifest. Paths and detailed receipt remain local.

This is **header/metadata validation**, not tensor-content validation or model
loading. Full weight hashes and remote revision are UNKNOWN; identity with the
historical checkpoint is UNKNOWN. No substitute Qwen2.5/quantized model is used.
Read-only device mapping on this resume exposes **one 910B2**, not TP4. Physical
mapping differs from the previous saved session; no framework/device workload
was started. Four-card allocation and compatible isolated runtime remain needed.

## Review gaps and next gate

`formal_m0_admission` in the manifest supplies explicit proposed splits,
correctness, B0/B1/B2/B3, metrics, budgets and stop rules. Proposed numbers are
not advisor-approved freezes. B3 eligibility, numerical tolerances, shape split,
service guardrails, exact container/model identity and four-card budget require
review. B2 is M1 work after M0, not something to implement to fill this form.

The critical decision: does the advisor accept a **clean prospective reproduction**
in one frozen current runtime as M0, given the historical provenance gaps?
Without that acceptance, current-runtime evidence is not pinned-container
reproduction. A repeat need not reproduce exactly +2.3%; the advisor must define
the accepted operator/service counterexample and attribution requirements.

Reuse the [TP4 plan](M0_TP4_PLAN_20261018.md); submit the
[review draft](M0_ISSUE20_REVIEW_DRAFT_20261018.md) manually after owner review.
No hardware, B2, framework installation or production edits in this update.

CPU validation on this update: existing probe 23 tests PASS; PR #14 attribution
13 cases, custody two cases and capture self-test PASS. Checkpoint checker also
has three CPU fixture tests (valid inventory, missing shard, truncated payload).
Large historical CSV custody remains NOT_RUN, not PASS.

Local receipts: `m0_receipts/m0_formal_admission_20261017/` (authority reads,
`host_checks_20261018/`, checkpoint inventory and CPU logs). Public summaries do
not contain raw tokens, device IDs or traces.
