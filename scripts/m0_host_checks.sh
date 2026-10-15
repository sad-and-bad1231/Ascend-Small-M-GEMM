#!/usr/bin/env bash
# Run checked-in PR#14's CPU-only M0 unit and self tests; never start serving.
# Usage: bash scripts/m0_host_checks.sh /absolute/pr14/checkout [local_output_dir]
set -euo pipefail
if [ "$#" -lt 1 ]; then echo "USAGE: $0 /path/to/pr14/checkout [local_output_dir]" >&2; exit 2; fi
root="$(cd "$1" && pwd -P)"
out="${2:-./m0_receipts/host_audit_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$out"
expected=11aca1de03129d6f4f56d846864af20ab35eaee8
actual="$(git -C "$root" rev-parse HEAD)"
if [ "$actual" != "$expected" ]; then
  echo "FAIL_CLOSED: PR#14 HEAD mismatch expected=$expected actual=$actual" >&2
  exit 3
fi
printf '%s\n' "$actual" > "$out/pr14_commit.txt"
git -C "$root" status --porcelain=v1 > "$out/pr14_dirty.local.txt"
for f in qkv_m0_attribution_test.py qkv_m0_verify_custody_test.py qkv_same_run_capture.py; do
  [ -f "$root/scripts/$f" ] || { echo "MISSING scripts/$f" >&2; exit 4; }
done
export PYTHONDONTWRITEBYTECODE=1
python3 -B "$root/scripts/qkv_m0_attribution_test.py" > "$out/attribution.log" 2>&1
python3 -B "$root/scripts/qkv_m0_verify_custody_test.py" > "$out/custody_unit.log" 2>&1
python3 -B "$root/scripts/qkv_same_run_capture.py" --self-test > "$out/capture_selftest.log" 2>&1
# Full custody hashing may require historical hundreds-of-MB CSVs. Opt in.
if [ "${M0_CHECK_CUSTODY:-0}" = 1 ]; then
  idx="$root/results/qkv_same_run_matched_20260827/inputs/custody_index.json"
  [ -f "$idx" ] || { echo "MISSING custody index" >&2; exit 5; }
  python3 -B "$root/scripts/qkv_m0_verify_custody.py" "$idx" > "$out/custody_full.log" 2>&1
else
  printf 'NOT_RUN: set M0_CHECK_CUSTODY=1 after all indexed CSV files are present\n' > "$out/custody_full.log"
fi
(cd "$out" && sha256sum *.log pr14_commit.txt pr14_dirty.local.txt > SHA256SUMS)
echo "M0_HOST_AUDIT_LOCAL_DIR=$out"
echo "Unit/self-tests completed. Historical raw custody status is in custody_full.log."
