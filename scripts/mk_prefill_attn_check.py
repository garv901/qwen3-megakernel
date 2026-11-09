"""MK_BIGT step 3: unit check of the prefill attention (kernels/mk/attn_prefill.cuh) outside the megakernel.

A test kernel (one 512-thread CTA per SM, the BIGT smem layout and barrier block) calls pf_attention for one layer; the output is compared with an
FP32 PyTorch causal GQA attention over the same bf16 q / K / V, and the call is timed (CUDA events, the item counter reset between calls).
PYTHONPATH=. python scripts/mk_prefill_attn_check.py [--cases A,B,C] [--iters 20] [--json out.json]"""
import argparse, json, math, sys
import torch
from regimebench.kernels.mk import build as B
from regimebench.kernels.mk.coop import Args, CoopKernel, i32, nvrtc_compile, parse_ptxas

KERNEL = r'''
extern "C" __global__ void __launch_bounds__(NT, 1) pf_test(const unsigned char* maps, const int4* items, int n_items, unsigned* ctr, bf16_t* ao, int cache_rows, int layer) {
    extern __shared__ unsigned char smraw[];
    const int tid = threadIdx.x, warp = tid >> 5;
    const u32 raw = smem_u32(smraw), base = (raw + 1023) & ~1023u;
    unsigned char* sm = smraw + (base - raw);
    volatile u32* tslot = reinterpret_cast<volatile u32*>(sm + OFF_MB + 128);
    if (warp == 0) mk_tmem_alloc(smem_u32((void*)tslot));
    if (tid == 0) {   // as the MK_BIGT prologue: every tile-stream mbarrier valid before pf_attention invalidates them
        for (int i = 0; i < 8; ++i) { mbar_init(base + OFF_MB + LMB_FULL + 8 * i, 2); mbar_init(base + OFF_MB + LMB_EMPTY + 8 * i, 1); }
        for (int b = 0; b < 2; ++b) { mbar_init(base + OFF_MB + LMB_ACCF + 8 * b, 1); mbar_init(base + OFF_MB + LMB_ACCE + 8 * b, 12); }
        mbar_fence_init();
        for (int m = 0; m < 3; ++m) map_acquire(maps + 128 * m);
    }
    __syncthreads();
    tc_fence_after();
    const u32 tb = *tslot;
    pf_attention(base, sm, base + OFF_MB, tb, maps, layer, cache_rows, items, n_items, ctr, ao);
    __syncthreads();
    tc_fence_after();
    if (warp == 0) mk_tmem_dealloc(tb);
}
'''

CASES = {
    "A": [(0, 1000)],                                       # one prefill, ragged last block
    "B": [(300, 200), (5000, 1), (0, 64), (77, 130), (0, 1)],   # mixed: history + chunk, long decode, exact block, ragged, single token
    "C": [(2048, 2048)],                                    # chunked prefill with history
    "D": [(0, 4096)],
    "E": [(0, 8192)],
    "F": [(0, 512)] * 8 + [(1000, 1)] * 32,                 # R7-like: 8 prefills + 32 decodes
}


def reference(seqs, qb, kc, vc, ntok):
    """FP32 causal GQA attention: out [ntok][16 * 128]."""
    out = torch.zeros(ntok, B.QH * B.HD, dtype=torch.float32, device=qb.device)
    tok, row = 0, 0
    sc = 1.0 / math.sqrt(B.HD)
    for kv, q in seqs:
        K = kc[row: row + kv + q].float()   # [P, 8, 128]
        V = vc[row: row + kv + q].float()
        for h in range(B.QH):
            kvh = h // 2
            for c0 in range(0, q, 1024):
                c1 = min(q, c0 + 1024)
                Q = qb[tok + c0: tok + c1, h].float()            # [n, 128]
                s = (Q @ K[:, kvh].T) * sc                      # [n, P]
                pos = torch.arange(kv + c0, kv + c1, device=qb.device)[:, None]
                s = s.masked_fill(torch.arange(kv + q, device=qb.device)[None, :] > pos, float("-inf"))
                out[tok + c0: tok + c1, h * B.HD: (h + 1) * B.HD] = torch.softmax(s, dim=-1) @ V[:, kvh]
        tok += q
        row += kv + q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="A,B,C,D,E,F")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.cuda.init()
    torch.zeros(1, device=dev)   # current context for the driver-API module load
    nb = torch.cuda.get_device_properties(dev).multi_processor_count
    src = "\n".join([B.ATTN_PLACEHOLDER.read_text(), *(open(B.HERE / f).read() for f in ("mk_common.cuh", "attn_prefill.cuh")), KERNEL])
    defs = ["NPAD=128", f"NB={nb}", *B.knob_defines({"MK_BIGT": 1, "MK_MAXT": 8192})]
    cubin, log = nvrtc_compile(src, defs, ptxas_verbose=True, name="pf_test.cu")
    dyn = B.dyn_smem(128, 128, 2, lmp=1, lmr=0, ts=128, tsb=1)
    kern = CoopKernel(cubin, "pf_test", dyn)
    print(f"pf_test: {parse_ptxas(log)} dyn {dyn} grid {nb}", flush=True)
    results, ok_all = {}, True
    for name in a.cases.split(","):
        seqs = CASES[name]
        torch.manual_seed(1234)
        ntok = sum(q for _, q in seqs)
        rows = sum(kv + q for kv, q in seqs)
        maxt = max(256, (ntok + 127) // 128 * 128)
        qb = torch.randn(maxt, B.QH, B.HD, device=dev).to(torch.bfloat16)
        kc = torch.zeros(rows + B.CTX, B.KVH, B.HD, dtype=torch.bfloat16, device=dev)
        vc = torch.zeros_like(kc)
        kc[:rows] = torch.randn(rows, B.KVH, B.HD, device=dev).to(torch.bfloat16)
        vc[:rows] = torch.randn(rows, B.KVH, B.HD, device=dev).to(torch.bfloat16)
        ao = torch.zeros(maxt, B.QH * B.HD, dtype=torch.bfloat16, device=dev)
        items = B.pf_plan(seqs)
        it = torch.tensor(items, dtype=torch.int32, device=dev)
        ctr = torch.zeros(B.NL, dtype=torch.int32, device=dev)
        maps = torch.frombuffer(bytearray(B.encode_pf_maps(qb.data_ptr(), kc.data_ptr(), vc.data_ptr(), rows, maxt, n_layers=1)), dtype=torch.uint8).to(dev)
        args = Args([maps.data_ptr(), it.data_ptr(), i32(len(items)), ctr.data_ptr(), ao.data_ptr(), i32(rows), i32(0)])
        st = torch.cuda.current_stream().cuda_stream
        kern.launch(nb, args, st)
        torch.cuda.synchronize()
        ref = reference(seqs, qb, kc.view(-1, B.KVH, B.HD), vc.view(-1, B.KVH, B.HD), ntok)
        got = ao[:ntok].float()
        err = (got - ref).abs()
        rel = float((got - ref).norm() / ref.norm())
        mx = float(err.max())
        bad = int((err > 0.02 + 0.02 * ref.abs()).sum())
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(a.iters):
            ctr.zero_()
            ev0.record()
            kern.launch(nb, args, st)
            ev1.record()
            torch.cuda.synchronize()
            ts.append(ev0.elapsed_time(ev1) * 1000.0)
        ts.sort()
        us = ts[len(ts) // 2]
        causal_pairs = sum(sum(kv + i + 1 for i in range(q)) for kv, q in seqs)
        tflops = 4.0 * causal_pairs * B.QH * B.HD / (us * 1e-6) / 1e12
        ok = rel < 1e-2 and bad == 0 and not math.isnan(mx)
        ok_all &= ok
        results[name] = dict(seqs=seqs, ntok=ntok, items=len(items), rel=rel, max_abs=mx, n_bad=bad, us_per_layer=us, us_step_28=us * 28, tflops=tflops, ok=ok)
        print(f"{name}: {'PASS' if ok else 'FAIL'} ntok {ntok} items {len(items)} rel {rel:.2e} max_abs {mx:.3e} bad {bad} | {us:.1f} us/layer ({us * 28 / 1000:.2f} ms/step) "
              f"{tflops:.0f} TFLOP/s (causal)", flush=True)
        del qb, kc, vc, ao, maps
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    print("ALL PASS" if ok_all else "SOME FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
