"""attn_ub: FlashInfer trtllm-gen decode kernel alone (the fi-graph decode-direct call, regimebench/arms/fi_graph.py _make_attn) at b x kv.

Paged KV (page 16, HND [pages, 2, 8, 16, 128] bf16), 16 q heads, one layer per call; L layers rotate so each call reads cold KV (>> L2 at kv >= 4K x b8).
Prints us/call (CUDA events, median of reps) and the achieved KV GB/s. Run under ncu for the kernel's launch config / DRAM counters.
Usage: fi_ub.py b8kv32768 [b8kv4096 ...] [--reps 30] [--layers 4]
"""
from __future__ import annotations

import argparse
import math
import re
import statistics

import torch
import flashinfer

PAGE, QH, KVH, D = 16, 16, 8, 128


def counter_buffer(batch: int, device="cuda") -> torch.Tensor:   # = fi_graph._counter_buffer
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    n = -(-max(batch * QH, sm) // 8) * 8
    return torch.zeros(n * 4, dtype=torch.uint8, device=device)


def run_point(name: str, reps: int, layers: int):
    b, kv = map(int, re.fullmatch(r"b(\d+)kv(\d+)", name).groups())
    pages = -(-kv // PAGE)
    kvl = [torch.randn(b * pages, 2, KVH, PAGE, D, dtype=torch.bfloat16, device="cuda") * 0.1 for _ in range(layers)]
    table = torch.arange(b * pages, dtype=torch.int32, device="cuda").view(b, pages)
    seq_lens = torch.full((b,), kv, dtype=torch.int32, device="cuda")
    q = torch.randn(b, QH, D, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(q)
    ws = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    cnt = counter_buffer(b)

    def call(i):
        flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q, kv_cache=kvl[i % layers], workspace_buffer=ws, block_tables=table, seq_lens=seq_lens, max_seq_len=kv,
            bmm1_scale=1.0 / math.sqrt(D), bmm2_scale=1.0, out=out, kv_layout="HND", enable_pdl=False, multi_ctas_kv_counter_buffer=cnt)

    for i in range(3):
        call(i)
    torch.cuda.synchronize()
    ts = []
    for i in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); call(i); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
    kv_bytes = b * kv * KVH * D * 2 * 2
    med = statistics.median(ts)
    n_g = 20                                                  # CUDA graph of n_g back-to-back calls (rotating layers): per-call time without launch latency
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for i in range(n_g):
                call(i)
    torch.cuda.synchronize()
    gs = []
    for _ in range(7):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record()
        torch.cuda.synchronize()
        gs.append(e0.elapsed_time(e1) * 1e3 / n_g)
    gm = statistics.median(gs)
    print(f"fi {name}: graph {gm:.2f} us/call -> {kv_bytes / gm / 1e3:.0f} GB/s", flush=True)
    print(f"fi {name}: {med:.2f} us/call (p10 {sorted(ts)[len(ts) // 10]:.2f}), KV {kv_bytes / 1e6:.1f} MB -> {kv_bytes / med / 1e3:.0f} GB/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points", nargs="+")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()
    for p in a.points:
        run_point(p, a.reps, a.layers)


if __name__ == "__main__":
    main()
