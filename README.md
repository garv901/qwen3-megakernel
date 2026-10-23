# regimebench

Measures the Qwen3-0.6B dense megakernel (hv2, ~383 us at decode b1 / kv 128 on B200) across serving
regimes (R0..R8, 45 points) against the best CUDA-graph + PDL multi-kernel baseline (FlashInfer / cuBLAS)
and vLLM, with a speed-of-light (SOL) number per point. See `docs/METHODOLOGY.md` and `SPEC.md`.

```
export PY=/raid/garv901/vibe-megakernel/.venv/bin/python PYTHONPATH=/raid/garv901/regime-megakernel
$PY -m regimebench run --arms fi-graph,hv2g,hv2-repeat,vllm --workloads R0,R1,R2-b1-kv8192 --tag NAME \
    [--seeds 7301,8917,42811] [--rounds 20] [--smoke] [--dry-run] [--variants "arm=v1,v2;..."]
$PY -m regimebench report runs/<RUN_DIR>... --out results/<date>-b200/     # REPORT.md, summary.csv, R*.png
sbatch slurm/regime.sbatch --arms ... --workloads ... --tag NAME          # GPU job (1 GPU, 2 h)
```

Layout: `regimebench/{geometry,workload,fixture}.py` (frozen inputs), `arms/` (fi_graph, hv2g, hv2_repeat,
vllm_worker), `ref/reference.py` (BF16 serving reference and FP32 oracle), `harness/` (timing, accuracy, JSONL
records), `sol.py`, `probes.py`, `run.py`, `report.py`, `scripts/gpu_check_*.py`, `tests/` (CPU pytest).
Records go to `runs/<UTC stamp>-<tag>/records.jsonl` (schema `regimebench/v1`); runs are never overwritten.
CPU tests: `$PY -m pytest -q tests/` (needs pytest on the path).
