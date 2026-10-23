# Minimal PR14 prospective adapter

This reuses PR14 `_load_prompts`, `_row_from_output`, `_validate_rows` and
`_aggregate`, plus the original LLM workload/Graph settings. It adds full token
custody, independent profile-off execution and bounded process cleanup. The
queued-input mode is a separate cohort, never pooled with ordinary generate.

Current state: **hardware stopped by failed full-output oracle**. Do not rerun
or unfreeze the manifest without a newly reviewed owner scope. No advisor
approval, production modification or package installation is implied.

Isolation used a local `git clone --shared` of installed plugin commit5cb98c,
with the existing `_version.py`, `_build_info.py`, compiled `.so`, `lib64`,
`include` and `_cann_ops_custom` build artifacts copied unchanged. Apply
`current_runtime.patch` only to that clone and verify with `git apply --check`
(or `--reverse --check` once applied). vLLM remains the read-only current
checkout. `launch.py` sets the isolated PYTHONPATH for driver and spawned ranks;
physical identifiers stay in local receipts. No full old submodule or CSV fetch.

The raw root must contain original32-of120 trace, checkpoint shard SHA256s,
runtime identity and `runtime/plugin` clone. Raw launch receipts contain exact
commands, source fingerprints and exit/cleanup state. Each label is unique;
existing attempts are never overwritten. `run_arm.py` is the internal child,
not an alternative permission entry point.

CPU-only reproduction of analysis, writing a **new** output path:

```bash
python experiments/m0tp4/analyze.py --raw m0_receipts/tp4_20261022 \
  --oracle-only --labels queued_B0_control1 queued_B0_control2 \
  --out m0_receipts/tp4_20261022/NEW_oracle_reanalysis.json
python -m unittest discover -s experiments/m0tp4 -p 'test_*.py'
```

Never use profiled rows as performance inputs. `analyze.py` rejects profiled
triplets, changed source identities, different full outputs and drift>=2%.
`--campaign-summary` preserves failed/partial attempts rather than silently
restricting its denominator to surviving arm receipts.
