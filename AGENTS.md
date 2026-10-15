# Execution Permissions and Evidence

- Read `docs/M0_PROTOCOL.md`, `docs/M0_MANIFEST.yaml` and the latest diagnostic
  report before continuing. Reuse existing environment/source receipts.
- Owner authorization to use an allocated NPU is sufficient for bounded local
  `DIAGNOSTIC_ONLY` smoke. Issue #20 advisor review governs FORMAL experiment
  admission, not each local hardware diagnostic. Never invent approval status.
- Honor the authorized shape, seed, stage, timeout and memory budget. Diagnostics
  retain source identity, correctness and fail-stop checks; they do not authorize
  formal timing sweeps, multi-card serving or M1 work.
- Short diagnostic profiling is allowed only for the owner's explicit manifest
  cells and replay cap; inspect each trace before expanding to the next arm/cell.
- Non-formal timing requires its own explicit owner-authorized manifest scope.
  Keep the one-triplet limit and all invalid data; never replace a drifting
  triplet with a favorable rerun or mix profiler measurements into timing.
- Preserve original failures before minimal harness repairs and reruns. Do not
  modify production kernels, golden vLLM/plugin checkouts or installed packages.
- Owner-requested updates may be committed and pushed to this personal notebook
  without another advisor gate. Never merge main or force-push automatically.
- Publish only sanitized summaries and checksums. Keep raw device logs, tensors,
  traces, identifiers and credentials local even when the owner calls this a
  private/personal repo; do not assume its hosting visibility is private.
