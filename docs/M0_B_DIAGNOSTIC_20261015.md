# M0-B Diagnostic Smoke

**DIAGNOSTIC_ONLY PASS; FORMAL admission remains false.**
Owner explicitly authorized the allocated single 910B2. Issue20 advisor review
is for formal experiment admission, not per-test local device permission.
The harness now distinguishes those modes. Diagnostics cannot select timing,
profiling, other shapes/seeds or non-chain scopes in this authorization.

## Actual Runs

Current M0-A source/runtime identity reused and checked; no packages, production
source, environment files or running services changed. Each final arm ran in a
fresh process, with the same code, tensors and startup flags. Matrix: BF16,
M=1, K=5120, N=2560, seed0, no bias; same existing fused consumer.

| Final run | Exit | Weight/output format | Eager + Graph oracle | Fixed replay / updated buffers | Difference vs first B0 |
| --- | --- | --- | --- | --- | --- |
| B0_attempt3 | 0 | ND2 / ND2 | PASS | 5 exact / PASS | reference |
| B1_attempt2 | 0 | NZ29 / ND2 | PASS | 5 exact / PASS | QKV/Q/K/V all exactly equal |
| B0_rollback_attempt1 | 0 | ND2 / ND2 | PASS | 5 exact / PASS | QKV/Q/K/V all exactly equal |

QKV vs CPU FP32: max abs **0.0106401**, relative L2 **0.00166295**.
Largest consumer/reference absolute error including updated inputs/positions:
**0.0237813**. All frozen elementwise and L2 criteria passed, V exact-copy,
eager/graph exact, all finite and expected shape/dtype/contiguity.
`max_rel` near zero can be large: pass is the declared
`abs(error) <= atol + rtol*abs(reference)`, not `max_rel <= rtol` alone.

After releasing the old weight, allocated bytes were **27,273,728 for both arms**;
final allocated **28,349,952**, reserved **54,525,952**. NZ load transient peak
was **54,536,704** versus ND **27,273,728**. These are Torch allocator values,
not total driver/context HBM. They are consistent with one resident weight, but
do not prove absence of internal GEMM conversions or transient hot-path copies.

**No performance Benchmark or profiler ran.** All timing trial arrays are empty,
median latency is null; setup wall-time diagnostics are not performance results.
This B0/B1/B0 validates numerical rollback, not latency drift or a speedup.
NZ storage plus successful F.linear does not establish a specific dispatch/kernel.
No formal M0/M1 or TP4 serving conclusion is implied.

## Preserved Failures and Minimal Fixes

1. `B0_attempt1`, exit1: the isolated consumer lacked Triton device properties.
   Added the existing `init_device_properties_triton()` startup call, which the
   current Ascend worker normally performs (`worker.py:501`). No kernel edit.
2. `B0_attempt2`, exit0: bootstrap validation after that fix. Retained, but not
   used in the final comparison because the next fix changes both arms' startup.
3. `B1_attempt1`, exit1: requested format29 was unavailable; runtime warned
   `allow_internal_format=False`. The numeric returned format was not logged.
   Added process-local `torch.npu.config.allow_internal_format=True` to **both**
   arms, as current production `model_runner_v1.py:201` already does. This permits
   internal storage; it does **not** enable global `weight_nz_mode=2` or cast
   non-QKV weights. Then reran the entire matched triplet with the same new probe.

Every failure stopped its process; logs and per-revision dirty patches remain
local, untouched. CPU validation after the final repair: py_compile exit0,
**10 unit/mock tests PASS**, including diagnostic/formal separation and both
startup prerequisites. Final device check: healthy, no running NPU processes.

## Reproduction and Custody

From base commit `79d451c`, final executed probe SHA256:
`897e8fb7a9fbbc12db2064237b0e33880ba0d248833fb31c66d5b1355c566237`.
The executed tracked dirty patch is retained as `executed_v3.patch`; its hash is
in the [sanitized summary](../results/m0b_diagnostic_20261015/summary.json).
All final results have identical input/source/runtime/probe fingerprints.

Actual final launch pattern (each command completed exit0 under a 120s timeout):

```bash
timeout 120s python experiments/m0b/qkv_probe.py --mode DIAGNOSTIC_ONLY --stage correctness --arm B0 --m 1 --seed 0 --out m0_receipts/m0b_diagnostic_20261015/B0_attempt3
timeout 120s python experiments/m0b/qkv_probe.py --mode DIAGNOSTIC_ONLY --stage correctness --arm B1 --m 1 --seed 0 --baseline m0_receipts/m0b_diagnostic_20261015/B0_attempt3/outputs.pt --out m0_receipts/m0b_diagnostic_20261015/B1_attempt2
timeout 120s python experiments/m0b/qkv_probe.py --mode DIAGNOSTIC_ONLY --stage correctness --arm B0 --m 1 --seed 0 --baseline m0_receipts/m0b_diagnostic_20261015/B0_attempt3/outputs.pt --out m0_receipts/m0b_diagnostic_20261015/B0_rollback_attempt1
```

Raw root: `m0_receipts/m0b_diagnostic_20261015/` (ignored). Contains progress and
console logs, CPU outputs, failure receipts, executed patches and device checks;
An additional compiler `fusion_result.json` appeared in the launch directory;
it was retained locally, not interpreted as dispatch/performance evidence.
`SHA256SUMS` indexes 33 local artifacts (hash pinned in manifest/summary).
Public/personal repo gets sanitized
summary, checksums and code only. Serialized B0/B1 output files have different
hashes because arm metadata differs, even though their tensor values match.

Next: retain this cell as a diagnostic baseline. Other shapes/seeds and formal
timing/profiling require their respective explicit scope/admission, not a forged
Issue20 approval. No further NPU work is included in this checkpoint.
