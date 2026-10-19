"""Repro: the G-Watch 0.0.35 cubin serializer fails on a global-to-shared bulk copy (SASS UBLKCP.S.G).

Usage: python tools/gwatch_serialize_repro.py   (CPU only; needs nvcc and the gwatch package)

The script compiles small sm_100 kernels, parses each cubin with gwatch, and serializes it again with no probe.
Expected output with gwatch 0.0.35: "ok" for every kernel except the one with UBLKCP.S.G.
"""
import os
import subprocess
import tempfile

import gwatch.cuda.binary as gw_binary

NVCC = os.environ.get("NVCC", "nvcc")

KERNELS = {
    "plain": """
extern "C" __global__ void k(float* a) { a[threadIdx.x] += 1.f; }""",
    "mbarrier only": """
extern "C" __global__ void k(float* a) {
  __shared__ alignas(8) unsigned long long mb; unsigned m = (unsigned)__cvta_generic_to_shared(&mb);
  if (threadIdx.x == 0) { asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(m));
                          asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(m)); }
  __syncthreads();
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared.b64 p, [%0], 0; @!p bra W; }" :: "r"(m));
  a[threadIdx.x] += 1.f; }""",
    "bulk copy shared -> global": """
extern "C" __global__ void k(float* a) {
  __shared__ alignas(128) float s[1024]; unsigned sm = (unsigned)__cvta_generic_to_shared(s);
  if (threadIdx.x == 0) { asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 4096;" :: "l"(a), "r"(sm) : "memory");
                          asm volatile("cp.async.bulk.commit_group; cp.async.bulk.wait_group 0;"); }
  a[threadIdx.x] += 1.f; }""",
    "bulk copy global -> shared": """
extern "C" __global__ void k(float* a) {
  __shared__ alignas(128) float s[1024]; __shared__ alignas(8) unsigned long long mb;
  unsigned sm = (unsigned)__cvta_generic_to_shared(s), m = (unsigned)__cvta_generic_to_shared(&mb);
  if (threadIdx.x == 0) { asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(m));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 4096;" :: "r"(m));
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 4096, [%2];"
                 :: "r"(sm), "l"(a), "r"(m) : "memory"); }
  __syncthreads();
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.shared.b64 p, [%0], 0; @!p bra W; }" :: "r"(m));
  a[threadIdx.x] = s[threadIdx.x] * 2.f; }""",
}


def main():
    with tempfile.TemporaryDirectory() as d:
        for name, src in KERNELS.items():
            cu, cubin = os.path.join(d, "k.cu"), os.path.join(d, "k.cubin")
            with open(cu, "w") as f:
                f.write(src)
            subprocess.run([NVCC, "-cubin", "-arch=sm_100", "-o", cubin, cu], check=True)
            sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True).stdout
            c = gw_binary.Cubin(); c.fill(cubin); c.parse()
            out = gw_binary.Cubin(); out.add_kerneldef(c.get_kerneldef_by_name("k"))
            try:
                res = f"ok ({len(out.serialize())} bytes)"
            except Exception as e:
                res = f"FAIL: {e}"
            print(f"{name:28s} UBLKCP.S.G={'UBLKCP.S.G' in sass!s:5s}  serialize: {res}")


if __name__ == "__main__":
    main()
