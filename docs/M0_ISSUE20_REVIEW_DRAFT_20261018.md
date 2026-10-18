# Issue #20 review draft - NOT POSTED

Target: https://github.com/intellistream/ascend-small-m-gemm/issues/20
This is a proposed owner comment, not advisor approval. formal_approved=false.

M0 admission preparation follows RESEARCH_CONTRACT and the 2026-10-14 handoff
at `cb095ea9dc335507ffdf7aedfb5d761fe6b5d88c`. Personal single-card NO-GO / MLP
transition is superseded. All failed/invalid results remain. Six seed0 screens:
3 INVALID, 1 valid +2.60%, 2 valid +12.43%/+12.21%; B1 only, not B2 or TP4 proof.

PR #14 and pinned fork objects are available. The original executed dirty patch,
container digest, checkpoint digest and full token outputs were not bound in
historical receipts. Sixteen large CSVs remain unverified; no bulk download.
Local BF16 Qwen3-32B has 17 structurally valid shards; weight-content hashes and
historical identity remain unverified. Only one 910B2 is allocated currently.

**Please review whether a clean prospective reproduction in an isolated, frozen
current runtime is acceptable for M0**, or identify the exact historical image
and checkpoint receipts required. This is not a request to compare arms across
versions. Please also define the accepted reproduction/attribution criterion
when historical exact execution identity cannot be recovered.

Proposed run reuses PR #14 capture and its ordered 32-request trace, B0/B1/B0,
64 greedy output tokens, TP4 FULL_DECODE_ONLY Graph sizes 1/2/4/8. Minimal
isolated adaptation must persist and compare full prompt/output IDs and text,
and separate unprofiled timing from profiling. Independent receipts bind source,
runtime, model, trace, outputs and device allocation. Global NZ is not B1.

Manifest `formal_m0_admission` proposes train/held-out and boundary shapes,
100% correctness, frozen tolerances/cosine, >=10 independent allocations for
future M1, B2 p95 >=5% below strongest eligible arm with all medians
non-regressing; future service primary TPOT >=2%, no extra full weight copy.
B3 eligibility is unresolved; B2 is not implemented and remains post-M0 work.
Please confirm split/tolerances/B3 eligibility and service guardrails.

Budget proposal: four exclusive 64GiB 910B2, initial one B0/B1/B0 panel <=60min,
<=1200s/arm, memory utilization 0.9, <=10GiB local traces. Five-triplet service
confirmation and separate profiling require explicit admission, not automatic
expansion. Rollback drift >=2%, any oracle/Graph/source/resource failure stop;
no favorable replacement. Failure to reproduce the accepted M0 counterexample
blocks M1. No production edits, environment upgrade, or B2 development now.

No formal launch until manifest approval, runtime compatibility, complete token
collector and four-card allocation are all satisfied. No comment has been sent.
