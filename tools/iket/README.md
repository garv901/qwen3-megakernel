# IKET trace extraction

These scripts made `docs/data/iket_trace.json` from the raw IKET trace (2.2 GB, not in this repository).
They need the trace decoder of our lab tools (`megabench.mkprobe.timeline.iket_adapter`), so they do not run outside the lab.

1. `decode.py OUT`: picks the launch with the median kernel time and writes the marker array `T[cta, layer, phase, marker, warp]`.
2. `breakdown.py NPZ JSON`: computes work, skew and hop for each phase.
3. `smid.py TRACE_DIR JSON`: writes the SM id of each CTA.
