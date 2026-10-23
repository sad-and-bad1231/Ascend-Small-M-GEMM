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
from stability_summary import variation


class ContractTests(unittest.TestCase):
    def test_screen_schedule_and_unselectable_gates(self):
        from paired_screen import ORDER, assess
        self.assertEqual(ORDER, ORDER[::-1])
        def rows():
            return [{"cycle": c, "slot": i, "arm": arm,
                     "event_us": 90 if arm == "B1" else 100,
                     "wall_us": 90 if arm == "B1" else 100}
                    for c in range(5) for i, arm in enumerate(ORDER)]
        r=rows()
        self.assertEqual(assess(r)["accepted_chain_gain_percent"], 10)
        r[1]["event_us"]=104
        self.assertEqual(assess(r)["status"], "INVALID_CONTROL")
        self.assertIsNone(assess(r)["accepted_chain_gain_percent"])
        r=rows(); r[3]["wall_us"]=102
        self.assertEqual(assess(r)["status"], "INVALID_CONTROL")
        with self.assertRaises(ValueError): assess(rows()[:-1])

    def test_screen_permission_m8_only_before_gate(self):
        import yaml
        manifest=yaml.safe_load((probe.ROOT/"docs/M0_MANIFEST.yaml").read_text())
        args=SimpleNamespace(stage="screen",mode="DIAGNOSTIC_ONLY",arm="B0",m=8,seed=0,
                             bias=False,scope="chain",physical_card=0,device_identity_sha="test")
        probe.require_execution_permission(manifest,args)
        for key,value in (("m",1),("seed",1),("mode","FORMAL"),("bias",True),("scope","gemm")):
            with self.assertRaises(RuntimeError):
                probe.require_execution_permission(manifest,SimpleNamespace(**{**vars(args),key:value}))
    def test_timeline_permission_is_separate_and_b0_only(self):
        import yaml
        manifest = yaml.safe_load((probe.ROOT / "docs/M0_MANIFEST.yaml").read_text())
        args = SimpleNamespace(mode="DIAGNOSTIC_ONLY", stage="timeline", scope="chain",
                               arm="B0", m=1, seed=0, bias=False, windows=1,
                               physical_card=0, device_identity_sha="test")
        probe.require_execution_permission(manifest, args)
        for key, value in (("arm", "B1"), ("m", 8), ("seed", 1), ("bias", True),
                           ("scope", "gemm"), ("windows", 3), ("mode", "FORMAL"),
                           ("physical_card", None), ("device_identity_sha", None)):
            with self.assertRaises(RuntimeError):
                probe.require_execution_permission(manifest, SimpleNamespace(**{**vars(args), key: value}))
        manifest["timeline_permission"]["user_authorized"] = False
        with self.assertRaises(RuntimeError):
            probe.require_execution_permission(manifest, args)

    def test_timeline_decimal_precision_and_overlapping_control(self):
        from decimal import Decimal
        from timeline_summary import timeline
        tasks, kernels = [], []
        origin = Decimal("1792070466970154.335")
        for i in range(5):
            start = origin + i * 100
            for kind, offset, duration, stream, tid in (
                    ("MODEL_EXECUTE", "0", "3", "47", str(81 + 2*i)),
                    ("AI_CORE", "2.94", "23.74", "13", "0"),
                    ("NOTIFY_WAIT", "14", "26", "47", str(82 + 2*i)),
                    ("AI_VECTOR_CORE", "26.72", "12.9", "13", "1"),
                    ("NOTIFY_RECORD", "39.64", "0.02", "13", "2")):
                name = "split_qkv_rmsnorm_rope_kernel" if kind == "AI_VECTOR_CORE" else "ND_GEMM" if kind == "AI_CORE" else "N/A"
                ts = start + Decimal(offset)
                tasks.append({"kernel_name": name, "kernel_type": kind, "stream_id": stream,
                              "task_id": tid, "task_start(us)": str(ts),
                              "task_stop(us)": str(ts + Decimal(duration)), "task_time(us)": duration})
                if kind in ("AI_CORE", "AI_VECTOR_CORE"):
                    kernels.append({"Start Time(us)": str(ts), "Stream ID": stream, "Task ID": tid,
                                    "Model ID": "49", "Name": name, "Block Num": "23",
                                    "Input Formats": "ND;ND", "Input Data Types": "DT_BF16;DT_BF16"})
        result = timeline(tasks, kernels)
        self.assertAlmostEqual(result["medians_us"]["handoff_gap_us"], 0.04)
        self.assertEqual(result["medians_us"]["graph_envelope_us"], 40)
        self.assertEqual(len(result["tasks_in_start_order"]), 25)
        self.assertEqual(result["medians_us"]["notify_wait_task_us"], 26)
        nz_kernels = [{**row, "Input Formats": "ND;FRACTAL_NZ"} for row in kernels]
        with self.assertRaises(ValueError):
            timeline(tasks, nz_kernels)
        self.assertEqual(timeline(tasks, nz_kernels, "FRACTAL_NZ")["medians_us"],
                         result["medians_us"])
        with self.assertRaises(ValueError):
            timeline(tasks[:-5], kernels)
    def test_stability_variation_all_pairs_and_boundary(self):
        result = variation([100, 102, 100])
        self.assertEqual(result["endpoint_drift_percent"], 0)
        self.assertTrue(result["blocks_ABA"])
        self.assertEqual(result["spread_over_first_percent"], 2)
        self.assertFalse(variation([100, 101, 100.5])["blocks_ABA"])
        for invalid in ([1, 2], [1, 0, 1], [1, float("nan"), 1]):
            with self.assertRaises(ValueError):
                variation(invalid)

    def test_stability_permission_narrow_and_separate(self):
        import yaml
        manifest = yaml.safe_load((probe.ROOT / "docs/M0_MANIFEST.yaml").read_text())
        args = SimpleNamespace(mode="DIAGNOSTIC_ONLY", stage="stability", scope="chain",
                               arm="B0", m=1, seed=0, bias=False, windows=3,
                               physical_card=2, device_identity_sha="test")
        probe.require_execution_permission(manifest, args)
        args.windows = 1
        probe.require_execution_permission(manifest, args)
        for key, value in (("arm", "B1"), ("m", 8), ("seed", 1), ("bias", True),
                           ("scope", "gemm"), ("windows", 2), ("mode", "FORMAL"),
                           ("physical_card", None), ("device_identity_sha", None)):
            with self.assertRaises(RuntimeError):
                probe.require_execution_permission(manifest, SimpleNamespace(**{**vars(args), key: value}))
        manifest["stability_permission"]["user_authorized"] = False
        with self.assertRaises(RuntimeError):
            probe.require_execution_permission(manifest, args)

    def test_device_identity_and_unexpected_process_stop(self):
        from stability_state import identity, validate
        board = "NPU ID : 2\nChip ID : 0\nChip Name : 910B2\nVDie ID : opaque\nPCIe Bus Info : opaque"
        fingerprint = identity(board)
        state = {"identity_sha256": fingerprint, "queries": {
            "board": {"exit_code": 0, "stdout": board},
            "mapping": {"exit_code": 0, "stdout": "2 0 0 Ascend 910B2"},
            "overview": {"exit_code": 0, "stdout": "OK Process id\nNo running processes found"}}}
        validate(state, fingerprint)
        with self.assertRaises(RuntimeError):
            validate(state, "changed")
        state["queries"]["overview"]["stdout"] = "OK Process id\n| 2 0 987654 python 100 987654 |"
        with self.assertRaises(RuntimeError):
            validate(state, fingerprint)

    def test_stability_identity_gate_precedes_framework_import(self):
        code = (
            "import runpy, sys, tempfile, pathlib, types\n"
            "def guard(event, args):\n"
            "    if event == 'import' and args[0].startswith(('torch', 'vllm')):\n"
            "        raise AssertionError('runtime import before identity gate')\n"
            "sys.addaudithook(guard)\n"
            "fake=types.ModuleType('stability_state')\n"
            "fake.snapshot=lambda card: {'identity_sha256': 'changed'}\n"
            "def reject(state, expected): raise RuntimeError('identity mismatch')\n"
            "fake.validate=reject\n"
            "sys.modules['stability_state']=fake\n"
            f"ns=runpy.run_path({probe.__file__!r})\n"
            "args=types.SimpleNamespace(mode='DIAGNOSTIC_ONLY',stage='stability',scope='chain',"
            "arm='B0',m=1,seed=0,bias=False,windows=1,physical_card=2,device_identity_sha='expected')\n"
            "manifest={'stability_permission': {'user_authorized': True},'timeline_permission': {'user_authorized': True},"
            "'approval': {'approved_for_hardware': False}}\n"
            "with tempfile.TemporaryDirectory() as tmp:\n"
            "    try: ns['hardware'](args,manifest,pathlib.Path(tmp))\n"
            "    except RuntimeError as exc: assert 'identity mismatch' in str(exc)\n"
            "    else: raise AssertionError('identity gate did not stop')\n"
            "    assert (pathlib.Path(tmp)/'pre_runtime_state.local.json').exists()\n"
        )
        for stage in ("stability", "timeline"):
            subprocess.run([sys.executable, "-S", "-c", code.replace("stage='stability'", f"stage='{stage}'")],
                           check=True, capture_output=True)

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
