"""GPU tests. They skip without CUDA.

Run from the repository root:  python -m unittest tests.test_gpu -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwen3mk import Megakernel, reference, weights  # noqa: E402

CUDA = torch.cuda.is_available()


def rel_l2(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a - b).norm() / b.norm())


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class KernelVsReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = weights.synthetic(0, "cuda")
        cls.mk = Megakernel()
        cls.out = cls.mk.step(cls.inputs)
        torch.cuda.synchronize()
        cls.ref = reference.decode_step(reference.to_fp32(cls.inputs), cls.inputs["token"],
                                        cls.inputs["kcache"], cls.inputs["vcache"])

    def test_next_token(self):
        self.assertEqual(int(self.out["next_token"]), int(self.ref["next_token"]))

    def test_logits(self):
        self.assertLessEqual(rel_l2(self.out["logits"], self.ref["logits"]), 1e-4)

    def test_kv_write(self):
        for n in ("k_write", "v_write"):
            self.assertLessEqual(float((self.out[n].double() - self.ref[n].double()).abs().max()), 0.02)

    def test_chained_launches(self):
        self.mk.bind(self.inputs, chained=True)
        for _ in range(5):
            self.mk.launch()
        torch.cuda.synchronize()
        self.assertTrue(0 <= int(self.mk.chain_token) < 151936)


@unittest.skipUnless(CUDA, "needs a CUDA GPU")
class ReferencePort(unittest.TestCase):
    """The port must match the MegaBench torch reference. Skips when megabench is not importable."""

    def test_matches_megabench_decode_step(self):
        sys.path.insert(0, "/raid/garv901/vibe-megakernel")
        try:
            from megabench.sota import qwen3_torch
            from megabench.sota.shapes import Shape
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"megabench not importable: {e}")
        inputs = weights.synthetic(1, "cuda")
        w = reference.to_fp32(inputs)
        mine = reference.decode_step(w, inputs["token"], inputs["kcache"], inputs["vcache"])
        state = qwen3_torch.build_state(Shape(1, 128), inputs, torch.float32, "cuda")
        theirs = qwen3_torch.decode_step(w, state, torch.float32)
        self.assertLessEqual(rel_l2(mine["logits"], theirs["logits"][0]), 1e-6)
        self.assertEqual(int(mine["next_token"]), int(theirs["next_token"][0]))


if __name__ == "__main__":
    unittest.main()
