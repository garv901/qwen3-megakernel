"""A0: descriptor unit test of the SKV_TC consumer (tc_unit.cu): S^T = K Q^T and O^T = V^T P^T vs torch. Usage: tc_unit.py [--swap] [--row0 N] [--kvh H]"""
import argparse, ctypes, struct, sys
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from regimebench.kernels.common.splitkv import header_source
from regimebench.kernels.hv2g.coop import Args, check, i32, nvrtc_compile

ap = argparse.ArgumentParser()
ap.add_argument("--swap", action="store_true", help="swap LBO / SBO of the V descriptor")
ap.add_argument("--row0", type=int, default=128)
ap.add_argument("--kvh", type=int, default=3)
ap.add_argument("--lbo", type=int, default=16384)
ap.add_argument("--sbo", type=int, default=1024)
ap.add_argument("--compile-only", action="store_true")
a = ap.parse_args()
src = header_source() + "\n" + (Path(__file__).with_name("tc_unit.cu")).read_text()
cubin, log = nvrtc_compile(src, ["SKV_TC=1"], arch="sm_100a", ptxas_verbose=True)
print(log)
if a.compile_only:
    sys.exit(0)
from cuda.bindings import driver as d
torch.cuda.init(); torch.zeros(1, device="cuda")
ROWS = 1024
g = torch.Generator(device="cuda"); g.manual_seed(1)
kc = (torch.randn(ROWS, 8, 128, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
vc = (torch.randn(ROWS, 8, 128, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
q = (torch.randn(16, 128, device="cuda", generator=g)).to(torch.bfloat16)
pt = torch.rand(16, 128, device="cuda", generator=g).to(torch.bfloat16)

def mk_map(t):
    res = d.cuTensorMapEncodeTiled(d.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, int(t.data_ptr()),
        [d.cuuint64_t(v) for v in (128, 8, ROWS)], [d.cuuint64_t(v) for v in (256, 2048)],
        [d.cuuint32_t(v) for v in (64, 1, 128)], [d.cuuint32_t(v) for v in (1, 1, 1)],
        d.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, d.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        d.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, d.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
    assert int(res[0]) == 0, res[0]
    return torch.frombuffer(bytearray(struct.pack("<16Q", *[int(v) for v in res[1].opaque])), dtype=torch.uint8).cuda()
km, vm = mk_map(kc), mk_map(vc)
mod = check(d.cuModuleLoadData(cubin)); fn = check(d.cuModuleGetFunction(mod, b"tc_unit"))
SM = 74000
check(d.cuFuncSetAttribute(fn, d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, SM))
sout = torch.zeros(128, 16, device="cuda"); oout = torch.zeros(128, 16, device="cuda")
lbo, sbo = (a.sbo, a.lbo) if a.swap else (a.lbo, a.sbo)
args = Args([km.data_ptr(), vm.data_ptr(), q.data_ptr(), pt.data_ptr(), sout.data_ptr(), oout.data_ptr(), i32(a.row0), i32(a.kvh), ctypes.c_uint(lbo), ctypes.c_uint(sbo), i32(8)])
check(d.cuLaunchKernel(fn, 1, 1, 1, 128, 1, 1, SM, torch.cuda.current_stream().cuda_stream, ctypes.addressof(args.ptrs), 0))
torch.cuda.synchronize()
K = kc[a.row0:a.row0 + 128, a.kvh].float(); V = vc[a.row0:a.row0 + 128, a.kvh].float()
Sref = K @ q.float().T            # [128 pos, 16]
Oref = V.T @ pt.float().T         # [128 dims, 16]
rs = float((sout - Sref).norm() / Sref.norm()); ro = float((oout - Oref).norm() / Oref.norm())
print(f"lbo {lbo} sbo {sbo}: S^T rel {rs:.3e}  O^T rel {ro:.3e}  ->", "PASS" if rs <= 1e-3 and ro <= 1e-3 else "FAIL")
if ro > 1e-3:
    print("O sample got", oout[:2, :4].tolist(), "ref", Oref[:2, :4].tolist())
sys.exit(0 if rs <= 1e-3 and ro <= 1e-3 else 1)
