# M0 execution checkpoint — 2026-10-15

Latest override (2026-10-22): **TP4 EXECUTED; HARDWARE STOPPED ON FULL-OUTPUT
ORACLE FAILURE**. See [all actual outcomes](M0_TP4_RESULTS_20261022.md).
Owner authorized local four-card prospective work, not formal advisor approval.
No accepted performance comparison or B2 follows the failed B0/B0 control.

Previous override (2026-10-18): **FORMAL M0 ADMISSION PREPARATION ONLY**.
Personal pause/MLP decisions and proposed additional M=1 screening are superseded.
Keep all historical measurements; no new seed0 screening, TP4 run or B2 development.
See [current readiness and gaps](M0_ADMISSION_REVIEW_20261018.md).

Status: **M8 REPEAT PANEL PASSES QUICK GATE / OVERALL REPRODUCIBILITY MIXED**.
Owner requested two more fixed rounds after228d971: four fresh M8 seed0 processes,
unchanged code/controls, all oracles pass. Added results: INVALID(raw-2.400%),
VALID2.599%, VALID12.425%, VALID12.206%. Round3 meets the original per-panel gate.
Across all rounds3/6 INVALID; no stable12% claim, formal GO or failure replacement.
M=1/integration were not run in repeat scope. See [all repeated results](M0_B_SCREEN_REPEATS_20261017.md).

Original final-screen checkpoint retained:
M=8 seed0 two-process paired screen completed: B0 ND, shadow ND and NZ all
correct,120 Event/Wall rows retained. Process1 shadow Event difference>=2%;
process2 Wall bracket drift2.196%. No accepted gain, M=1 gate not met.
No integration/fusion development or further stability experiments. This is
measurement blocking, not NZ/fusion falsification.23 CPU tests pass; formalfalse.
See [final GO/NO-GO decision](M0_B_FINAL_SCREEN_20261017.md).

Earlier timeline checkpoint:
2026-10-17 current-card B0-only M=1 seed0 panel completed: three fresh processes,
each unprofiled ->5 profiled replays ->unprofiled. All oracles pass;21 CPU tests.
Consumer medians18.32/24.76/14.80us, GEMM23.22/23.90/24.12us, handoff0.02-0.04us.
Device Graph envelopes track Event variation; before/after drift never reaches2%.
Missing flow links and absent counters leave underlying cause and exact
unprofiled attribution UNKNOWN. No B1, A/B/A or optimization benefit claim.
See [device tasks and evidence boundary](M0_B_TIMELINE_20261017.md).

Earlier stability checkpoint:
Owner subsequently authorized an independent batch on the newly allocated card.
Four processes exit0, six windows/30 trials, full/exact oracles pass. Same-Graph
Event spread0.47%; fresh-process spread14.77%, endpoint drift10.01%.
Host enqueue spans do not rise commensurately; root cause UNKNOWN. Cold start
Event-record/query paths account for most first-Wall excess, not Event drift.
No B1, profiler, retry, formal approval or production edit.19 CPU tests pass.
See [new batch and preserved prior incident](M0_B_STABILITY_20261015.md).

Earlier allocation-change checkpoint:
B0-only stability scope was prepared and statically reviewed, but resume on
2026-10-16 exposed a different physical card. One erroneously launched process
was aborted (exit143), zero measurement windows, three fresh controls NOT_RUN.
Identity validation now precedes framework import;18 CPU tests pass. See
[stability review and preserved incident](M0_B_STABILITY_20261015.md).
No new stability or optimization result is claimed; A/B/A remains blocked.

Earlier timing checkpoint:
Owner-authorized M1/M8 seed0 chain timing completed, one fresh-process B0/B1/B0
triplet each. Full oracles passed, but Event rollback drift was 32.41% / 4.84%.
No retries or accepted performance gain. See [all trials and boundaries](M0_B_TIMING_PILOT_20261015.md).
Formal approval remains false; no M1/M2 GO follows from this diagnostic.

Earlier dispatch checkpoint:
M1/M8 seed0 short traces observe distinct ND/NZ GEMM native kernels and the same
consumer, no standalone conversion task in five replays/arm, exact output pairs.
M16/bias fallback pairs pass. See [dispatch evidence and limitations](M0_B_DISPATCH_20261015.md).
At that checkpoint unprofiled timing had not run; the pilot was only a proposal.

Earlier smoke checkpoint:
Owner-authorized M=1 seed0 B0/B1/B0 correctness and Graph Replay passed on the
allocated 910B2. See [diagnostic evidence and preserved failures](M0_B_DIAGNOSTIC_20261015.md).
No performance benchmark, TP4 capture or profiler ran in that earlier smoke.
The M0-A and CPU-only preparation records below are historical checkpoints.

M0-B preparation based on personal main `c2fdf300`: see the
[single-card review contract](M0_B_CONTRACT_20261015.md) and manifest. The new
isolated probe defaults to a CPU-only plan; FORMAL runtime entry is advisor-gated,
while bounded DIAGNOSTIC_ONLY correctness smoke uses explicit owner authorization.
No environment re-inventory or NPU execution in this preparation. The numbered
M0-A execution order below is retained history, not an instruction to rerun it.

## Independently captured M0-A

- Personal main synchronized at `633ea9943d34f534fae9b102244d17c00000e457`;
  upstream main/PR14 remote refs match the frozen SHAs below.
- Actual environment: one visible Ascend910B2, driver26.0.rc1, CANN9.1.0;
  Python3.12.13, torch2.10.0+cpu, torch-npu2.10.0.post4;
  vLLM0.23.0+empty / Ascend0.23.0, source `0fc695f` / `5cb98ca`, clean.
- `m0_preflight.sh` exit0; three `m0_host_checks.sh` CPU/self-tests exit0.
  CANN/driver version files supplemented the script's legacy version.cfg probes.
- Shared-object sparse PR14 checkout is clean and about412KiB; no large CSV
  downloaded/materialized. Twelve indexed small inputs passed size/SHA checks;
  six request windows re-aggregate identically. Sixteen CSVs remain unverified.
- Old NZ patch checks against pinned files but fails on the current plugin.
  Qwen3 attention forward AST and the 910B2 consumer kernel are unchanged;
  fusion registration/loader paths differ. Runtime match/dispatch NOT_RUN.
- See [acceptance and source drift](M0_A_ACCEPTANCE_20261015.md) for receipts,
  the initial checkout bootstrap failure, unverified files and remaining gates.

## Original user-reported preflight (retained historical checkpoint)

- Python `3.12.13`; PyTorch `2.10.0+cpu`; `torch_npu 2.10.0.post4`.
- `torch.npu.is_available()=True`; visible NPU count `1`.
- Active vLLM checkout `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` (v0.23.0 line).
- Warning: `LD_PRELOAD` / FunctionLoader — warning only, not a diagnosed failure.
- **Unknown at initial drafting:** device/driver, CANN, container digest,
  plugin checkout/patch identity, model/trace identity. The audit above resolves
  device/software identity; container digest and target model/trace remain unknown.
- Official Ascend compatibility table pairs `torch_npu 2.10.0.post4` with PyTorch `2.10.0` and CANN `9.1.0`; `+cpu` in PyTorch is intentional, not a mismatch. The table is *not* proof this container runs CANN 9.1.0: https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md.

## Source and custody checkpoints

- Organization research contract: `intellistream/ascend-small-m-gemm@cb095ea9dc335507ffdf7aedfb5d761fe6b5d88c`.
- Historical **closed, unmerged** PR #14: `11aca1de03129d6f4f56d846864af20ab35eaee8`.
- The PR #14 branch, *not* current upstream main, contains:
  - `scripts/qkv_m0_attribution_test.py`
  - `scripts/qkv_m0_verify_custody_test.py`
  - `scripts/qkv_same_run_capture.py --self-test`
  - `scripts/qkv_m0_verify_custody.py <custody_index.json>`
  - `results/qkv_same_run_matched_20260827/inputs/` (large historical CSV evidence).
- The old capture driver defaults to `--npus 0,1,2,3`, requires pinned fork submodules, a pre-applied policy patch, and a Qwen3-32B model path. **Never execute its online orchestrator on the one-device machine.**
- Historical 2026-08-27 capture lacks a complete executable source/dirty patch receipt and output-token equivalence; those gaps cannot be retrospectively repaired. See [research state](RESEARCH_STATE.md).

## Immediate execution order

1. Run `bash scripts/m0_preflight.sh` locally on the Coder instance. Retain the generated directory locally; do not publish device identifiers/host/container logs verbatim.
2. Use a separate checkout at exact PR #14 SHA. Prefer a partial/filtered checkout to avoid immediately materializing the historical hundreds-of-MB `kernel_details.csv` files. Worktree must preserve commit identity.
3. Run `bash scripts/m0_host_checks.sh /absolute/path/to/PR14-checkout`. Save exit codes and logs locally. The tests are CPU/static only. **Do not report old tests as passing until actual new logs exist.**
4. Compare the actual clean status of the installed `vllm` and `vllm_ascend` to the pinned historical forks, identify the real QKV->consumer graph path and available source/patch. Treat version drift as a new treatment domain, not as identical August-2026 replication.
5. Populate [M0 manifest](M0_MANIFEST.yaml), especially installed CANN, container digest, source/patch hashes, exact oracle tolerance, budget, and reviewer approval.
6. Submit the complete manifest to [Issue #20](https://github.com/intellistream/ascend-small-m-gemm/issues/20) for advisor approval **before formal** M0-B experiments. Explicit owner-authorized bounded diagnostics are separate and do not confer formal admission.

## Hardware gate

- **M0-B:** on 1×910B2, matched per-rank QKV `X[M,5120]`, `W[2560,5120]`, BF16, `M=1,2,4,8`, plus M16 fallback. Compare B0 official QKV + current *existing* fused consumer versus B1 genuine load-time QKV-only NZ + exactly the same consumer, with graph replay and complete output oracle. Attribute GEMM/consumer/chain and memory/conversions using NPU events/profiler, retaining all raw allocations.
- **M0-C:** requires 4 compatible NPUs, actual Qwen3-32B TP4 model and a matched B0/B1/B0 service A/B/A panel with token-ID equality. **Do not call single-rank synthetic timings a TP4 service result.**
- **M1:** only after a valid M0 and reviewer gate, implement a genuinely new native producer-consumer primitive. Do not switch research topic just because the *old* B1 mechanism fails.

## Public-repo discipline

This personal repo is public by owner choice. Commit only sanitized manifests, logic, bibliographies and reviewed summaries. Keep raw profiler traces, device/host IDs, proprietary model assets, credentials and detailed runtime receipts in a separate controlled location. Never commit a false `PASS`.
