# M0 execution checkpoint — 2026-10-15

Status: **M0-A HOST CHECKS PASS / CUSTODY PARTIAL / M0-B BLOCKED**.
Host audit recorded on 2026-10-15 (Asia/Shanghai). No new NPU correctness,
timing, TP4 capture or profiler result is asserted; hardware approval remains false.

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
6. Submit the complete manifest to [Issue #20](https://github.com/intellistream/ascend-small-m-gemm/issues/20) for advisor approval **before** M0-B hardware timing. A partial record is NOT hardware approval.

## Hardware gate

- **M0-B:** on 1×910B2, matched per-rank QKV `X[M,5120]`, `W[2560,5120]`, BF16, `M=1,2,4,8`, plus M16 fallback. Compare B0 official QKV + current *existing* fused consumer versus B1 genuine load-time QKV-only NZ + exactly the same consumer, with graph replay and complete output oracle. Attribute GEMM/consumer/chain and memory/conversions using NPU events/profiler, retaining all raw allocations.
- **M0-C:** requires 4 compatible NPUs, actual Qwen3-32B TP4 model and a matched B0/B1/B0 service A/B/A panel with token-ID equality. **Do not call single-rank synthetic timings a TP4 service result.**
- **M1:** only after a valid M0 and reviewer gate, implement a genuinely new native producer-consumer primitive. Do not switch research topic just because the *old* B1 mechanism fails.

## Public-repo discipline

This personal repo is public by owner choice. Commit only sanitized manifests, logic, bibliographies and reviewed summaries. Keep raw profiler traces, device/host IDs, proprietary model assets, credentials and detailed runtime receipts in a separate controlled location. Never commit a false `PASS`.
