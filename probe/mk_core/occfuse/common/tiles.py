"""Tile arithmetic (B9): for a GEMM phase and a design, the work each CTA gets. No GPU.
Weights W [N][K] (bf16, M = 128 rows per block), tokens T. A k-step = 128 k = 32 KB of W per 128-row block.
Designs:
  classic(N, K, grid, ku=128)              split-K units (128 rows x ku k), contiguous ranges over all CTAs (mk geom())
  ts(N, K, T, TT, KS, grid)                tiles (rb, tt, ks), contiguous ranges (mk ts_geom()), each tile = K/KS/128 k-steps
Returns: units/tiles, busy CTAs, max and mean k-steps per CTA, max weight bytes per CTA, idle CTAs, outputs written per tile.
Usage: tiles.py N K T TT KS [grid=148]   (prints classic and ts side by side)"""
import sys


def _ranges(total, grid):
    return [(c + 1) * total // grid - c * total // grid for c in range(grid)]


def classic(N, K, grid=148, ku=128):
    nrb, kc = N // 128, K // ku
    u = _ranges(nrb * kc, grid)
    steps = [x * ku // 128 for x in u]
    return dict(design="classic", units=nrb * kc, busy=sum(1 for x in u if x), idle=sum(1 for x in u if not x),
                max_steps=max(steps), mean_steps=sum(steps) / grid, max_w_bytes=max(steps) * 32768)


def ts(N, K, T, TT, KS, grid=148):
    nrb, ntt = N // 128, max(1, -(-T // TT))
    nt = nrb * ntt * KS
    spt = K // (KS * 128)
    t = _ranges(nt, grid)
    return dict(design="ts", tiles=nt, busy=sum(1 for x in t if x), idle=sum(1 for x in t if not x), max_tiles=max(t),
                steps_per_tile=spt, max_steps=max(t) * spt, mean_steps=nt * spt / grid, max_w_bytes=max(t) * spt * 32768)


if __name__ == "__main__":
    N, K, T, TT, KS = map(int, sys.argv[1:6])
    g = int(sys.argv[6]) if len(sys.argv) > 6 else 148
    print(classic(N, K, g)); print(ts(N, K, T, TT, KS, g))
