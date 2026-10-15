#!/usr/bin/env bash
# M0 environment identification only; no model execution or benchmarking.
# All receipts are local and may contain host identifiers. DO NOT publish raw.
set -euo pipefail
out="${1:-./m0_receipts/preflight_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  uname -a
  printf 'container_digest=%s\n' "${CONTAINER_IMAGE_DIGEST:-UNAVAILABLE}"
  printf 'CANN_HOME=%s\n' "${ASCEND_HOME_PATH:-UNSET}"
} > "$out/identity.local.txt"
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info > "$out/npu_smi.local.txt" 2>&1 || true
else
  printf 'npu-smi MISSING\n' > "$out/npu_smi.local.txt"
fi
{
  for f in /usr/local/Ascend/ascend-toolkit/latest/version.cfg /usr/local/Ascend/cann/version.cfg; do
    if [ -r "$f" ]; then
      printf 'VERSION_FILE=%s\n' "$f"
      head -30 "$f"
    fi
  done
  command -v ccec || true
  command -v msprof || true
} > "$out/cann.local.txt" 2>&1
python3 -B - > "$out/python.local.txt" 2>&1 <<'PY'
import sys, importlib.util
print("python:", sys.version)
try:
    import torch, torch_npu
    print("torch:", torch.__version__)
    print("torch_npu:", torch_npu.__version__)
    print("npu_available:", torch.npu.is_available())
    print("npu_count:", torch.npu.device_count())
    if torch.npu.is_available():
        print("npu_name:", torch.npu.get_device_name(0))
except Exception as e:
    print("IMPORT_OR_DEVICE_QUERY_FAILURE:", repr(e))
    raise
for name in ("vllm", "vllm_ascend", "triton"):
    spec = importlib.util.find_spec(name)
    print(name, "origin:", None if spec is None else spec.origin)
PY
{
  for p in /vllm-workspace/vllm "${SMALL_M_VLLM_ROOT:-}" "${SMALL_M_ASCEND_ROOT:-}" "${SMALL_M_ORG_ROOT:-}"; do
    [ -n "$p" ] && [ -d "$p/.git" ] || continue
    echo "== $p =="
    git -C "$p" rev-parse HEAD
    git -C "$p" status --porcelain=v1
    git -C "$p" submodule status || true
  done
} > "$out/git.local.txt" 2>&1
(cd "$out" && sha256sum *.local.txt > SHA256SUMS)
echo "M0_PREFLIGHT_LOCAL_DIR=$out"
echo "PREFLIGHT_ONLY: not a benchmark, not NPU correctness, not advisor approval."
