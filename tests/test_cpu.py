"""CPU tests. They need no GPU.

Run from the repository root:  python -m unittest tests.test_cpu -v

The SASS hash gate pins the machine code of csrc/qwen3_step.cu. The hash was
equal to the hash of the original experiment kernel when this repository was made.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import sass_hash  # noqa: E402
from qwen3mk import kernel  # noqa: E402

# SASS sha of the pruned kernel (NVRTC, sm_100, -DNB=148). It equals the sha of the original source.
EXPECTED_SASS_SHA = "a4454be9373e5d21521a36079d3b91069e62261b1db300ca6b95798376811c07"
EXPECTED_REGS = 126
EXPECTED_STATIC_SMEM = 47568

# The original experiment module. Tests that compare against it skip when it is absent.
HV2_PATH = Path(os.environ.get(
    "HV2_SUBMISSION",
    "/raid/garv901/vibe-megakernel/megabench/mkprobe/runs/20261012T232636Z-HV2-host/src/hv2_submission.py"))


def load_hv2():
    spec = importlib.util.spec_from_file_location("hv2_submission", HV2_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class KernelSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = sass_hash.analyze(kernel.source_text())

    def test_compiles_for_sm100(self):
        self.assertIsNotNone(self.result["regs"])

    def test_no_spill(self):
        self.assertEqual(self.result["spill"], 0)

    def test_regs_within_budget(self):
        self.assertLessEqual(self.result["regs"], 128)
        self.assertEqual(self.result["regs"], EXPECTED_REGS)

    def test_static_smem(self):
        self.assertEqual(self.result["static_smem"], EXPECTED_STATIC_SMEM)

    def test_sass_sha(self):
        self.assertEqual(self.result["sass_sha"], EXPECTED_SASS_SHA)

    def test_no_experiment_switches_left(self):
        src = kernel.source_text()
        for token in ("#ifndef", "#if ", "#ifdef", "R10_", "R11_", "R9_"):
            self.assertNotIn(token, src)


@unittest.skipUnless(HV2_PATH.exists(), "original hv2 module not found")
class AgainstHv2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hv2 = load_hv2()

    def test_dyn_smem(self):
        self.assertEqual(kernel.DYN_SMEM_BYTES, self.hv2._dyn_smem())

    def test_scratch_and_sync(self):
        self.assertEqual(kernel.SCRATCH_FLOATS, self.hv2._scratch_floats())
        self.assertEqual(kernel.SYNC_INTS, self.hv2._SYNC_INTS)
        self.assertFalse(self.hv2._skd_on())  # no tensor map is needed

    def test_input_order(self):
        self.assertEqual(kernel.INPUT_ORDER, self.hv2._INPUT_ORDER)

    def test_same_sass_as_original(self):
        # The original source with the compile options of hv2 `_compile` for sm_100.
        orig = sass_hash.analyze(self.hv2.KERNEL_SRC)
        new = sass_hash.analyze(kernel.source_text())
        self.assertEqual(orig, new)


class Prune(unittest.TestCase):
    def test_prune_reproduces_the_shipped_kernel(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import prune
        if not HV2_PATH.exists():
            self.skipTest("original hv2 module not found")
        out = prune.prune(load_hv2().KERNEL_SRC, public=True)
        self.assertEqual(out, kernel.source_text())

    def test_evaluator(self):
        import prune
        src = "#ifndef A\n#define A 2\n#endif\n#if A == 2 && !defined(B)\nlive\n#else\ndead\n#endif\n#if NB > 1\nkeep\n#endif\n"
        out = prune.prune(src)
        self.assertIn("live", out)
        self.assertNotIn("dead", out)
        self.assertIn("#if NB > 1", out)  # NB comes from the command line: undecidable, kept


class Host(unittest.TestCase):
    def test_rope_table_shape(self):
        import torch
        t = kernel.rope_table("cpu")
        self.assertEqual(tuple(t.shape), (128,))
        self.assertEqual(t.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
