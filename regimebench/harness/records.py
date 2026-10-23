"""JSONL result records, run directories, and environment capture.

Every record is one JSON line with ``schema``, ``run_id``, ``record_type``,
``timestamp_utc`` and ``slurm_job_id``. Files are exclusive-create and are
never overwritten; a crashed run keeps whatever was flushed.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

SCHEMA = "megabench.sota/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(__file__).resolve().parent / "runs"
ENV_PREFIXES = ("FLASHINFER_", "TORCHINDUCTOR_", "VLLM_")
ENV_NAMES = ("HF_HUB_OFFLINE", "HF_HOME", "TRITON_CACHE_DIR", "MEGABENCH_SOTA_CACHE")


class RecordWriter:
    """Append-only JSONL writer; the file is created exclusively on first use."""

    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self._file = None

    def _open(self):
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "x")
        return self._file

    def write(self, record_type: str, **fields) -> dict:
        record = {"schema": SCHEMA, "run_id": self.run_id,
                  "record_type": record_type,
                  "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        record.update(fields)
        self.append(record)
        return record

    def append(self, record: dict) -> None:
        """Write an already-formed record unchanged (used to merge worker files)."""
        f = self._open()
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def read_records(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def new_run_dir(tag: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / f"{stamp}-{tag}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, cwd=cwd)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except Exception:
        return None


def env_info() -> dict:
    """Host, GPU, package versions (without importing them), git and env vars."""
    info: dict = {"host": socket.gethostname()}
    try:
        import torch
        info["torch_cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": p.name, "sm_count": p.multi_processor_count,
                           "l2_bytes": getattr(p, "L2_cache_size", None),
                           "memory_bytes": p.total_memory,
                           "capability": list(torch.cuda.get_device_capability(0))}
    except Exception as exc:
        info["torch_error"] = repr(exc)
    info["driver"] = _run(["nvidia-smi", "--query-gpu=driver_version",
                           "--format=csv,noheader"])
    if info["driver"]:
        info["driver"] = info["driver"].splitlines()[0]
    info["versions"] = {d: _version(d) for d in
                        ("torch", "flashinfer-python", "vllm", "triton",
                         "nvidia-ml-py")}
    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        lines = [x for x in nvcc.splitlines() if x.strip()]
        rel = [x for x in lines if "release" in x or x.startswith("Build")]
        nvcc = (rel or lines)[-1] if lines else None
    info["nvcc"] = nvcc
    sha = _run(["git", "rev-parse", "HEAD"], REPO_ROOT)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], REPO_ROOT)
    info["git"] = {"sha": sha, "dirty": bool(status) if status is not None else None}
    info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    info["SLURM_JOB_ID"] = os.environ.get("SLURM_JOB_ID")
    info["env"] = {k: v for k, v in sorted(os.environ.items())
                   if k.startswith(ENV_PREFIXES) or k in ENV_NAMES}
    return info
