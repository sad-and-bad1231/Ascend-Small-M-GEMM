"""CPU-only preparation tests. No import of torch_npu or the production consumer."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import qkv_probe as probe
from trace_summary import trace_events
from timing_summary import assess


class ContractTests(unittest.TestCase):
    def test_timing_gain_and_exact_drift_boundary(self):
        trials = [[{"event_us": v, "wall_us": v} for _ in range(5)] for v in (100, 90, 100)]
        summary = assess(trials)
        self.assertEqual(summary["status"], "PILOT_DRIFT_SCREEN_PASS")
        self.assertEqual(summary["metrics"]["event_us"]["raw_latency_reduction_percent"], 10)
        trials[2] = [{"event_us": 102, "wall_us": 100} for _ in range(5)]
        summary = assess(trials)
        self.assertEqual(summary["status"], "INVALID_DRIFT")
        self.assertEqual(summary["metrics"]["event_us"]["rollback_absolute_drift_percent"], 2)
        self.assertIsNone(summary["metrics"]["event_us"]["drift_screened_point_estimate_percent"])

    def test_wall_drift_also_invalidates_and_no_trial_filter(self):
        trials = [[{"event_us": v, "wall_us": v} for _ in range(5)] for v in (100, 90, 100)]
        trials[2] = [{"event_us": 100, "wall_us": 102} for _ in range(5)]
        self.assertEqual(assess(trials)["status"], "INVALID_DRIFT")
        trials[0].pop()
        with self.assertRaises(ValueError):
            assess(trials)

    def test_trace_array_and_object_envelopes(self):
        events = [{"name": "HostToDevice1", "ph": "f"}, {"name": "MODEL_EXECUTE", "ph": "X"}]
        self.assertEqual(trace_events(events), events)
        self.assertEqual(trace_events({"traceEvents": events}), events)

    def test_isolated_runtime_initializes_worker_prerequisites(self):
        npu = SimpleNamespace(config=SimpleNamespace(allow_internal_format=False))
        utils = SimpleNamespace(init_device_properties_triton=Mock())
        probe.initialize_isolated_runtime(npu, utils)
        self.assertTrue(npu.config.allow_internal_format)
        utils.init_device_properties_triton.assert_called_once_with()

    def test_policy(self):
        for m in (1, 2, 4, 8):
            self.assertEqual(probe.weight_format("B1", m), 29)
            self.assertEqual(probe.weight_format("B0", m), 2)
            self.assertEqual(probe.weight_format("B1", m, True), 2)
        self.assertEqual(probe.weight_format("B1", 16), 2)
        for m in (0, 3, 32):
            with self.assertRaises(ValueError):
                probe.weight_format("B1", m)
        with self.assertRaises(ValueError):
            probe.weight_format("B1", 1, shape=(5120, 2560))

    def test_gate(self):
        for value in (False, "true", None):
            with self.assertRaises(RuntimeError):
                probe.require_approval({"approval": {"approved_for_hardware": value}})
        with self.assertRaises(RuntimeError):
            probe.require_approval({"approval": {"approved_for_hardware": True, "advisor_review_link": "TO_FILL"}})

    def test_diagnostic_permission_is_not_formal_approval(self):
        manifest = {"diagnostic_permission": {"user_authorized": True, "cells": [
                        {"m": 1, "bias": False, "stages": ["correctness", "profile"]},
                        {"m": 8, "bias": False, "stages": ["correctness", "profile"]},
                        {"m": 16, "bias": False, "stages": ["correctness"]},
                        {"m": 1, "bias": True, "stages": ["correctness"]}]},
                    "approval": {"approved_for_hardware": False, "advisor_review_link": "TO_FILL"}}
        args = SimpleNamespace(mode="DIAGNOSTIC_ONLY", stage="correctness", scope="chain",
                               m=1, seed=0, bias=False)
        probe.require_execution_permission(manifest, args)
        self.assertFalse(manifest["approval"]["approved_for_hardware"])
        for key, value in (("stage", "timing"), ("scope", "gemm"),
                           ("m", 2), ("seed", 1), ("bias", True), ("mode", "FORMAL")):
            altered = SimpleNamespace(**{**vars(args), key: value})
            if key == "bias":
                altered.stage = "profile"
            with self.assertRaises(RuntimeError):
                probe.require_execution_permission(manifest, altered)
        manifest["diagnostic_permission"]["user_authorized"] = False
        with self.assertRaises(RuntimeError):
            probe.require_execution_permission(manifest, args)

    def test_diagnostic_profiles_and_fallback_cells(self):
        import yaml
        manifest = yaml.safe_load((probe.ROOT / "docs/M0_MANIFEST.yaml").read_text())
        for m, bias, stage in ((1, False, "profile"), (8, False, "profile"),
                               (16, False, "correctness"), (1, True, "correctness")):
            args = SimpleNamespace(mode="DIAGNOSTIC_ONLY", stage=stage, scope="chain",
                                   m=m, seed=0, bias=bias)
            probe.require_execution_permission(manifest, args)
            if m == 16 or bias:
                args.stage = "profile"
                with self.assertRaises(RuntimeError):
                    probe.require_execution_permission(manifest, args)

    def test_bounded_timing_permission(self):
        import yaml
        manifest = yaml.safe_load((probe.ROOT / "docs/M0_MANIFEST.yaml").read_text())
        for m in (1, 8):
            args = SimpleNamespace(mode="DIAGNOSTIC_ONLY", stage="timing", scope="chain",
                                   m=m, seed=0, bias=False)
            probe.require_execution_permission(manifest, args)
            for key, value in (("m", 2), ("m", 16), ("seed", 1), ("bias", True),
                               ("scope", "gemm"), ("scope", "consumer"), ("mode", "FORMAL")):
                altered = SimpleNamespace(**{**vars(args), key: value})
                with self.assertRaises(RuntimeError):
                    probe.require_execution_permission(manifest, altered)
        self.assertFalse(manifest["approval"]["approved_for_hardware"])
        self.assertFalse(manifest["timing_pilot_permission"]["formal_approved"])

    def test_plan_stdlib_only(self):
        result = subprocess.run([sys.executable, "-S", str(Path(probe.__file__)), "--stage", "plan"],
                                check=True, text=True, capture_output=True)
        self.assertEqual(len(json.loads(result.stdout)["cells"]), 6)
        tree = ast.parse(Path(probe.__file__).read_text())
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any("torch" in ast.unparse(n) for n in imports))

    def test_hardware_gate_before_runtime_import(self):
        code = (
            "import runpy, sys\n"
            "def guard(event, args):\n"
            "    if event == 'import' and args[0].startswith(('torch', 'vllm')):\n"
            "        raise AssertionError('runtime import before approval')\n"
            "sys.addaudithook(guard)\n"
            f"ns = runpy.run_path({probe.__file__!r})\n"
            "try:\n"
            "    ns['hardware'](None, {'approval': {'approved_for_hardware': False}}, None)\n"
            "except RuntimeError as exc:\n"
            "    assert 'BLOCKED' in str(exc)\n"
            "else:\n"
            "    raise AssertionError('gate did not stop hardware path')\n"
        )
        subprocess.run([sys.executable, "-S", "-c", code], check=True, capture_output=True)

    def test_oracle_identity_rope_and_v(self):
        # Tiny hand-checkable geometry: q=2 heads, kv=1 head, dim=4.
        qkv = torch.arange(1, 17, device="cpu").bfloat16()[None, :]
        cache = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], device="cpu").bfloat16()
        w = torch.ones(4, device="cpu").bfloat16()
        q, k, v = probe.consumer_reference(qkv, cache, torch.tensor([0], device="cpu"), w, w,
                                          head_dim=4, q_heads=2, kv_heads=1)
        expected = qkv[:, :8].float().reshape(1, 2, 4)
        expected = (expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)).reshape(1, 8)
        probe.compare(q, expected, atol=0, rtol=0)
        probe.compare(v, qkv[:, 12:].float(), atol=0, rtol=0)
        qr, kr, _ = probe.consumer_reference(qkv, cache, torch.tensor([1], device="cpu"), w, w,
                                            head_dim=4, q_heads=2, kv_heads=1)
        a, b = q.reshape(1, 2, 4).chunk(2, -1)
        probe.compare(qr, torch.cat((-b, a), -1).reshape(1, 8), atol=0, rtol=0)
        a, b = k.chunk(2, -1)
        probe.compare(kr, torch.cat((-b, a), -1), atol=0, rtol=0)

    def test_oracle_zero_and_target_geometry(self):
        qkv = torch.zeros(2, 2560, device="cpu", dtype=torch.bfloat16)
        cache = torch.zeros(4096, 128, device="cpu", dtype=torch.bfloat16)
        w = torch.ones(128, device="cpu", dtype=torch.bfloat16)
        result = probe.consumer_reference(qkv, cache, torch.tensor([0, 4095], device="cpu"), w, w)
        self.assertEqual([tuple(t.shape) for t in result], [(2, 2048), (2, 256), (2, 256)])
        self.assertTrue(all(torch.equal(t, torch.zeros_like(t)) for t in result))

    def test_reject_bad_outputs(self):
        a = torch.ones(4, device="cpu")
        for b in (a * float("nan"), a * float("inf"), a + 1, torch.ones(5, device="cpu")):
            with self.assertRaises(AssertionError):
                probe.compare(a, b)
        with self.assertRaises(AssertionError):
            probe.compare(a, a + 0.02, atol=1, rel_l2_limit=0.01)

    def test_manifest_plan_agrees(self):
        import yaml
        manifest = yaml.safe_load((probe.ROOT / "docs/M0_MANIFEST.yaml").read_text())
        self.assertFalse(manifest["approval"]["approved_for_hardware"])
        component = manifest["component_preparation"]
        self.assertEqual(component["matrix"], probe.plan()["cells"])
        self.assertEqual(component["consumer"]["sha256"], probe.CONSUMER_SHA)
        self.assertEqual(manifest["correctness"]["consumer_atol"], probe.ATOL)
        self.assertEqual(manifest["correctness"]["consumer_rtol"], probe.RTOL)


if __name__ == "__main__":
    unittest.main()
