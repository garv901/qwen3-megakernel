// bar.cu: grid-barrier microbench (SPEC-bar). One arm per build (-DBAR=n). 148 CTAs x 512 threads, cooperative persistent launch.
// Usage: ./bin/bar<n> [nit=20000] [reps=5] [skews="0,100"]   (JSON lines on stdout, PASS/FAIL on stderr)
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;
#ifndef BAR
#define BAR 0
#endif
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
typedef unsigned long long ull;
constexpr int NT = 512;
constexpr int SMEM = 100 * 1024;
constexpr int NGRP = 8;
// sync-word buffer layout, in 128-B lines (32 words each)
constexpr int L_CNT = 0, L_FLAG = 1, L_GRP = 2 /* 8 lines */, L_TOP = 10, L_VIOL = 11, L_ARR = 12 /* 5 lines: 148 packed words */, L_SLOT = 20 /* 5 lines */;
constexpr int NLINES = 32;

__device__ __forceinline__ ull gt() { ull t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ unsigned ld_acq(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ unsigned ld_rlx(const unsigned* p) { unsigned v; asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void st_rel(unsigned* p, unsigned v) { asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_rlx(unsigned* p, unsigned v) { asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ unsigned atom_rel(unsigned* p, unsigned v) { unsigned o; asm volatile("atom.add.release.gpu.global.u32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory"); return o; }
__device__ __forceinline__ unsigned atom_acqrel(unsigned* p, unsigned v) { unsigned o; asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], %2;" : "=r"(o) : "l"(p), "r"(v) : "memory"); return o; }

struct Bar {
    unsigned* cnt; unsigned* flag; unsigned* grp; unsigned* top; unsigned* arr;
    unsigned gsize, grpid;
    unsigned target;      // arms 0/1: running target (mk style)
    bool last;            // arms 2/6: this CTA was the last arriver (thread 0 / warp 0 only)
};
__device__ __forceinline__ void bar_init(Bar& B, unsigned* w) {
    B.cnt = w + L_CNT * 32; B.flag = w + L_FLAG * 32; B.grp = w + L_GRP * 32; B.top = w + L_TOP * 32; B.arr = w + L_ARR * 32;
    const unsigned G = gridDim.x, b = blockIdx.x;
    B.grpid = b * NGRP / G;
    const unsigned first = (B.grpid * G + NGRP - 1) / NGRP, nxt = ((B.grpid + 1) * G + NGRP - 1) / NGRP;
    B.gsize = nxt - first;
    B.target = 0; B.last = false;
}

#if BAR == 0 || BAR == 1
// arm 0 = mk v0 bar_arrive / bar_wait (mk_common.cuh), no stamps. arm 1: acquire poll, no separate fence.
__device__ __forceinline__ void bar_arrive(Bar& B, unsigned) {
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(B.cnt) : "memory");
}
__device__ __forceinline__ void bar_wait(Bar& B, unsigned) {
    B.target += gridDim.x;
    if (threadIdx.x == 0) {
        unsigned v;
        do {
#if BAR == 0
            asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(B.cnt) : "memory");
#else
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(B.cnt) : "memory");
#endif
        } while (v < B.target);
#if BAR == 0
        asm volatile("fence.acquire.gpu;" ::: "memory");
#endif
    }
    __syncthreads();
}
#elif BAR == 2
__device__ __forceinline__ void bar_arrive(Bar& B, unsigned epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = atom_acqrel(B.cnt, 1);   // acq_rel (spec: release): the last arriver must acquire the others' writes
        B.last = (old == epoch * gridDim.x - 1);
        if (B.last) st_rel(B.flag, epoch);
    }
}
__device__ __forceinline__ void bar_wait(Bar& B, unsigned epoch) {
    if (threadIdx.x == 0 && !B.last) { while (ld_acq(B.flag) != epoch) {} }
    __syncthreads();
}
#elif BAR == 3
__device__ __forceinline__ void bar_arrive(Bar& B, unsigned epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned og = atom_acqrel(B.grp + B.grpid * 32, 1);
        if (og == epoch * B.gsize - 1) {
            const unsigned ot = atom_acqrel(B.top, 1);   // acq_rel (spec: release): the top-last must acquire the other groups' writes
            if (ot == epoch * NGRP - 1) st_rel(B.flag, epoch);
        }
    }
}
__device__ __forceinline__ void bar_wait(Bar& B, unsigned epoch) {
    if (threadIdx.x == 0) { while (ld_acq(B.flag) != epoch) {} }
    __syncthreads();
}
#elif BAR == 4
__device__ __forceinline__ void bar_arrive(Bar& B, unsigned epoch) {
    __syncthreads();
    if (threadIdx.x == 0) st_rel(B.arr + blockIdx.x, epoch);
}
__device__ __forceinline__ void bar_wait(Bar& B, unsigned epoch) {
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        const unsigned G = gridDim.x;
        bool ok;
        do {
            ok = true;
            for (unsigned k = threadIdx.x; k < G; k += 32) ok &= (ld_acq(B.arr + k) == epoch);
        } while (!__all_sync(0xffffffffu, ok));
        if (threadIdx.x == 0) st_rel(B.flag, epoch);
    }
    if (threadIdx.x == 0) { while (ld_acq(B.flag) != epoch) {} }
    __syncthreads();
}
#elif BAR == 5
__device__ __forceinline__ void bar_arrive(Bar&, unsigned) {}
__device__ __forceinline__ void bar_wait(Bar&, unsigned) { cg::this_grid().sync(); }
#elif BAR == 6
__device__ __forceinline__ void bar_arrive(Bar& B, unsigned epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned old = atom_acqrel(B.cnt, 1);
        B.last = (old == epoch * gridDim.x - 1);
        if (B.last) st_rel(B.flag, epoch);
    }
}
__device__ __forceinline__ void bar_wait(Bar& B, unsigned epoch) {
    if (threadIdx.x < 32) {
        const bool last = __shfl_sync(0xffffffffu, (int)B.last, 0);
        if (!last) { while (!__any_sync(0xffffffffu, ld_acq(B.flag) == epoch)) {} }
    }
    __syncthreads();
}
#else
#error "BAR must be 0..6"
#endif

template <bool VAL>
__global__ void __launch_bounds__(NT, 1) k_bar(unsigned* w, int nit, unsigned skew_ns, ull* tm) {
    extern __shared__ __align__(1024) unsigned char smem[];
    if (nit < 0) smem[threadIdx.x] = 1;   // never taken; keeps the dynamic smem reference
    Bar B; bar_init(B, w);
    unsigned* slot = w + L_SLOT * 32;
    const unsigned G = gridDim.x, b = blockIdx.x;
    const ull dwait = (ull)(b % 8) * skew_ns;
    ull c0 = 0, g0 = 0;
    if (threadIdx.x == 0) { c0 = clock64(); g0 = gt(); }
    unsigned viol = 0;
    for (int i = 0; i < nit; ++i) {
        const unsigned epoch = (unsigned)i + 1;
        if (dwait && threadIdx.x == 0) { const ull t0 = gt(); while (gt() - t0 < dwait) {} }
        if (VAL && threadIdx.x == 0) st_rlx(slot + b, epoch);
        bar_arrive(B, epoch);
        bar_wait(B, epoch);
        if (VAL && threadIdx.x == 0) { if (ld_rlx(slot + (b + 37u * (unsigned)i) % G) < epoch) ++viol; }
    }
    if (threadIdx.x == 0) {
        const ull c1 = clock64(), g1 = gt();
        ull* o = tm + b * 4; o[0] = c0; o[1] = c1; o[2] = g0; o[3] = g1;
        if (VAL && viol) atomicAdd(w + L_VIOL * 32, viol);
    }
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    int nit = argc > 1 ? atoi(argv[1]) : 20000;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    std::vector<unsigned> skews;
    { const char* s = argc > 3 ? argv[3] : "0,100"; while (*s) { skews.push_back((unsigned)strtoul(s, (char**)&s, 10)); if (*s == ',') ++s; } }
    int dev = 0; CK(cudaGetDevice(&dev));
    int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
    const int G = nsm;
    if (G != 148) fprintf(stderr, "warning: %d SMs (expected 148)\n", G);
    CK(cudaFuncSetAttribute(k_bar<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    CK(cudaFuncSetAttribute(k_bar<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    int occ = 0, occv = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_bar<false>, NT, SMEM));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occv, k_bar<true>, NT, SMEM));
    if (occ < 1 || occv < 1) { fprintf(stderr, "occupancy %d/%d < 1, abort\n", occ, occv); return 1; }
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_bar<false>));
    unsigned* w; ull* tm;
    CK(cudaMalloc(&w, NLINES * 128)); CK(cudaMalloc(&tm, G * 4 * sizeof(ull)));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto launch = [&](bool val, int n, unsigned skew) {
        CK(cudaMemset(w, 0, NLINES * 128));
        int nn = n; unsigned sk = skew;
        void* args[] = {&w, &nn, &sk, &tm};
        CK(cudaLaunchCooperativeKernel(val ? (void*)k_bar<true> : (void*)k_bar<false>, dim3(G), dim3(NT), args, SMEM, 0));
    };
    // validation (skew 0 and 100)
    for (unsigned sk : {0u, 100u}) {
        launch(true, 2000, sk); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        unsigned viol = 0; CK(cudaMemcpy(&viol, w + L_VIOL * 32, 4, cudaMemcpyDeviceToHost));
        if (viol == 0) fprintf(stderr, "PASS bar=%d skew_ns=%u\n", BAR, sk); else fprintf(stderr, "FAIL violations=%u bar=%d skew_ns=%u\n", viol, BAR, sk);
    }
    launch(false, 200, 0); CK(cudaDeviceSynchronize());   // warmup
    for (unsigned sk : skews) {
        std::vector<double> us, mhz;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(e0)); launch(false, nit, sk); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); us.push_back(ms * 1000.0 / nit);
            std::vector<ull> h(G * 4); CK(cudaMemcpy(h.data(), tm, G * 4 * sizeof(ull), cudaMemcpyDeviceToHost));
            mhz.push_back((double)(h[1] - h[0]) / (double)(h[3] - h[2]) * 1000.0);   // CTA 0: cycles per ns * 1000
        }
        printf("{\"bar\":%d,\"skew_ns\":%u,\"us_per_bar\":%.4f,\"us_reps\":[", BAR, sk, median(us));
        for (size_t i = 0; i < us.size(); ++i) printf("%s%.4f", i ? "," : "", us[i]);
        printf("],\"mhz\":%.0f,\"G\":%d,\"regs\":%d}\n", median(mhz), G, fa.numRegs);
        fflush(stdout);
    }
    return 0;
}
