# M0-B: Single-Card Contract for Review

**PREPARATION ONLY / CPU TESTED / HARDWARE NOT RUN / NOT APPROVED.**
The status above describes the original preparation checkpoint. Subsequent
owner-authorized `DIAGNOSTIC_ONLY` smoke is separate from formal admission;
see the diagnostic section in the manifest. Issue20 review gates **FORMAL**
experiments, not each bounded local diagnostic. No approval is fabricated.
Based on personal main `c2fdf300`. Reuse [M0-A audit](M0_A_ACCEPTANCE_20261015.md):
one 910B2, vLLM `0fc695f`, Ascend plugin `5cb98ca`, current 0.23 stack.
No environment re-inventory, package changes, production patches, TP4 runner,
new kernel, serving or device allocation in this preparation.
Machine-readable choices: [manifest](M0_MANIFEST.yaml).

## 1. Executability and Equivalence

| Item | Source-confirmed contract | Remaining device evidence |
| --- | --- | --- |
| Producer B0 | Current `ops/linear.py::unquantized_gemm` is `F.linear(X,W,bias)`; explicit ND weight | Actual ND dispatch and output format |
| Producer B1 | Same `F.linear`, only this synthetic QKV weight load-cast to FRACTAL_NZ=29; release ND weight before capture | NZ lowering works on current stack, no hidden hot conversion/second resident weight |
| Consumer | Exactly the same installed `split_qkv_rmsnorm_rope_impl` callable/file for both arms | Compile, eager correctness, capture and replay |
| Graph | Installed `torch_npu/npu/graphs.py` has NPUGraph, replay, `auto_dispatch_capture=True` | API presence is NOT runtime capability; failure stops timing, no eager substitution |
| Old implementation | Reuse PR14 `qkv_capture_policy.py` eligibility and `tinyprojbench_graph.py` F.linear/NZ/event method | Do not execute old environment activation, old TP4 runner, or apply incompatible loader patch |

This is a **synthetic TP4-rank-shaped component**, not full Qwen3 TP1 or TP4
serving. It bypasses scheduler, attention, collectives, weight loader and compiler
pattern matching; calling the existing consumer directly does NOT prove current
vLLM fusion dispatch. M0-A already identified the compiler-to-consumer source path.
Both arms use ONLY the current software domain; historical numbers are not an arm.

Both arms regenerate identical CPU tensors from the same seed and save input hashes.
Separate sequential processes avoid holding ND and NZ weights/graphs simultaneously.
B1 requires a B0 output receipt matching input hashes, source SHAs, package versions
and probe hash. No global `weight_nz_mode=2`, no transposed duplicate weight,
no activation repair `.contiguous()`/format cast between producer and consumer.
Consumer output allocations are preserved identically, not optimized away.

## 2. Real Layout and Geometry

The [official Qwen3-32B config at frozen revision
9216db5](https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json)
has hidden_size=5120, head_dim=128, Q heads=64, KV heads=8, no attention bias,
eps=1e-6, rope_theta=1e6. With TP4: 16 Q heads and 2 KV heads per rank,
so **2048+256+256=2560**. Do NOT infer head_dim as 5120/64=80.
Only the small public config was fetched; this is not verification of the historical
checkpoint identity. No weights downloaded or loaded.

The installed consumer is byte-identical to the pinned fork (hash in manifest).
Kernel offsets are `row * total_hidden_size + column`, **not input strides**.
Require BF16 physical ND contiguous `[M,2560]`, strides `[2560,1]`.
Weight NZ is fine only if GEMM produces this ordinary ND activation.
Consumer returns separate contiguous BF16 `Q[M,2048], K[M,256], V[M,256]`.
Q/K RMSNorm is per 128-element head in FP32, then BF16 norm rounding before
NeoX split-half RoPE; V is copied exactly. Norm weights each have 128 elements.
Consumer norm-bias arguments stay `None`; the bias fallback below is **GEMM bias**.

Use full128 rotary dimension, BF16 cache `[4096,128]` = cos-half then sin-half,
positions `row*257` int64 and a replay +1 update. It is a compact synthetic position
panel, not a captured decode trace or full 40960-position model cache.

## 3. Frozen Matrix and Oracle

| M | GEMM bias | B0 weight | B1 policy | Purpose |
| --- | --- | --- | --- | --- |
| 1,2,4,8 | none | ND=2 | NZ=29 | Four primary cells |
| 16 | none | ND=2 | ND=2 | Out-of-policy fallback, not an NZ test |
| 1 | BF16 `[2560]` | ND=2 | ND=2 | Bias fallback, not norm bias |

All use BF16 `X[M,5120]`, `W[2560,5120]`. CPU-seeded random X/norm weights,
W scaled by `1/sqrt(K)` before BF16 conversion. They model shape, not real
checkpoint distributions. Seeds 0..9 are frozen; no seed selection after results.

**Fail any check => retain failure, stop the cell before timing.**

1. Check finite, all shapes, BF16 dtype, contiguity and physical ND producer output.
2. QKV vs CPU FP32 GEMM on the exact BF16 operands: all elements within
   `atol=.05, rtol=.01` AND relative L2 <=.01. Historical TinyProjBench only used
   rel_L2<=.01; the elementwise guard is a new preregistered proposal, not an
   upstream/pinned GEMM tolerance. No post-result widening.
3. Consumer vs independent CPU FP32 split/RMSNorm/NeoX reference on **its own QKV**:
   `atol=.05, rtol=.005`, from byte-identical pinned/current op tests. That test
   covers other head counts and M=1/16, not this exact M=2/4/8,16Q/2KV panel.
4. Check the chain against FP32 GEMM rounded to BF16 then independent reference;
   compare B1 QKV/Q/K/V to the matching B0 receipt under the respective tolerances.
   V must equal its own input slice exactly. Fallback arms must match B0 exactly.
5. Eager vs graph exact outputs; five fixed-buffer replays exact; mutate X and
   positions in place, replay and re-check oracle, then restore before measurement.

The CPU reference deliberately keeps normalization in FP32 like the pinned test;
the kernel rounds intermediate norm to BF16. Tolerance is intentional, not
bitwise agreement with an accidentally identical reimplementation. Token-ID
equality remains an M0-C requirement, not something this synthetic test can prove.

## 4. Timing, Profiling and Budget

After approval: first **M=1, seed0, correctness-only B0/B1/B0**, then M=16 and
M=1+bias fallback. If valid, progress M=2/4/8 correctness; only then timing.
Formal primary panel: 10 independent fresh allocations (seeds0..9) for each
primary M, each in sequential **B0/B1/B0** processes. Fixed shapes, no padding.
Fallback smokes are descriptive controls, not 10-repeat speed claims.

Per graph: 3 eager warmups, 10 warmup replays, 5 event/wall trials x100 replays.
Retain every trial; report allocation median, p5/p95 **of allocation medians**,
paired B1/B0 ratio and rollback drift. Five trial batches are not five independent
allocations, and this p95 is not a per-launch tail latency. If rollback baseline
median drift is >=2%, invalidate the triplet; do not cherry-pick reruns.

Chain replay is primary. Separate GEMM-only and fixed-ND-input consumer probes
are diagnostics, not additive estimates of chain critical path. Consumer-only
receives identical B0 QKV from the paired baseline receipt rather than each
arm's numerically slightly different output.
Same shape/code alone cannot establish consumer hardware causality.

Profile separately, M=1 and M=8 seed0, at most 5 replays/arm. Save raw timeline,
dispatch ID, launch counts, conversion and allocator evidence locally; review
hot-path weight/activation format conversions and memory. GEMM/consumer names
alone are not an attribution. GM/MTE/UB/AIC/AIV wait counters are optional: missing
counters or unresolved dispatch => **UNKNOWN**, not a fabricated bottleneck.

One-time weight cast wall time, before/after allocation, transient load peak,
capture/replay allocated/reserved/peak are recorded separately. Logical weight
is **26,214,400 bytes =25MiB**. A cast transiently overlaps allocations and NZ may
pad storage: no-copy means **no second resident weight**, NOT zero allocation or
zero format cost. Current runtime implicit costs remain UNKNOWN until tracing.

Proposed total device budget: **1x910B2, <=30 minutes wall occupancy, <=2GiB allocated
HBM, <=1GiB local traces**, 120s timeout per process (including first compilation).
Operator checks ownership/idle state with `npu-smi info` and stores its output
locally before allocation; do not kill someone else's process. Stop at total
budget, first oracle/layout/capture failure, source/runtime drift, or unsupported
NZ. Incomplete repeats are descriptive-only. Hidden repeated conversion or an
extra resident weight invalidates genuine B1 admission, even if timing is faster.
No single-card result grants M0-C/M1 GO.

## 5. Prepared Entry Points and Review Gate

Safe now (CPU only):

```bash
python -S experiments/m0b/qkv_probe.py --stage plan
python -m py_compile experiments/m0b/qkv_probe.py experiments/m0b/test_qkv_probe.py
python -m unittest discover -s experiments/m0b -p 'test_*.py' -v
```

The original FORMAL commands below are **templates, NOT executed in preparation**. First obtain an
explicit advisor Issue20 approval comment and update the reviewed manifest
`approval.approved_for_hardware` and its exact `advisor_review_link`.
No FORMAL bypass is provided. Separately, `--mode DIAGNOSTIC_ONLY --stage correctness`
allows the explicitly owner-authorized M=1 seed=0 no-bias chain cell without an
advisor link; it rejects timing/profile stages, other shapes/seeds and scopes.
Source identity, correctness, budget and fail-stop checks are unchanged.
Reuse the existing allocated device visibility;
do not hard-code yesterday's physical device ID. With an idle/owned device:

```bash
timeout 120s python experiments/m0b/qkv_probe.py --stage correctness --arm B0 --m 1 --seed 0 --out m0_receipts/REVIEWED_RUN/B0
timeout 120s python experiments/m0b/qkv_probe.py --stage correctness --arm B1 --m 1 --seed 0 --baseline m0_receipts/REVIEWED_RUN/B0/outputs.pt --out m0_receipts/REVIEWED_RUN/B1
timeout 120s python experiments/m0b/qkv_probe.py --stage correctness --arm B0 --m 1 --seed 0 --baseline m0_receipts/REVIEWED_RUN/B0/outputs.pt --out m0_receipts/REVIEWED_RUN/A2
```

Later use `--stage timing` for unprofiled trials or `--stage profile` for a separate
short trace, same controls. `--scope gemm|consumer` is diagnostic only.
Each invocation is one fresh allocation; no sweep/orchestrator added. Save launch
command, current git SHA/dirty patch, output, exit code and checksums locally, even
on timeout/failure. New output directories cannot overwrite old receipts.

## 6. Preparation Acceptance and Blockers

- `py_compile`: exit0; stdlib-only `--stage plan`: exit0; eight CPU unit tests:
  exit0. Includes an import-audited negative hardware gate test with approval
  false, plus oracle/policy/fallback/nonfinite/manifest tests. None are device
  correctness, graph capture or performance PASS. The familiar LD_PRELOAD warning
  appeared during CPU torch import; no new environment diagnosis is claimed.
- Hardware path is prepared but **unvalidated**. PyTorch/plugin/Triton ABI,
  output physical format, no-copy dispatch and capture compatibility are open.
- Container digest remains unknown from M0-A; record an actual digest or obtain
  explicit reviewer acceptance of this identity gap. No silent invented identity.
- Public Issue20/API returned HTTP404 during preparation; no current approval
  could be verified. This does not imply the issue is absent or rejected.
  Manifest approval stays false. Do not run hardware until its approval is supplied.
- Historical custody gaps, dirty-runner provenance and unverified CSVs remain as
  M0-A recorded; no new large downloads or retroactive PASS.

Preparation receipts stay in `m0_receipts/m0b_preparation_20261015/` (ignored).
Only scripts, manifest and this sanitized contract are public.
