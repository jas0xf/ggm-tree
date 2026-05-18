"""Measurement runner: warmup + repeats per (backend, prg, kernel, depth) cell."""

from __future__ import annotations
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ggm.tree import GGMTree


@dataclass
class BenchResult:
    backend: str
    prg: str
    kernel: str
    key_mode: str
    threads: int
    depth: int
    mode: str
    median_seconds: float
    iqr_seconds: float
    min_seconds: float
    leaves_per_sec: float
    bytes_per_sec: float
    repeats: int
    warmup: int
    seed_hex: str
    git_sha: str
    cpu_model: str
    gpu_model: str


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _gpu_model() -> str:
    try:
        import pycuda.driver as drv  # type: ignore
        import pycuda.autoinit  # type: ignore  # noqa: F401

        return drv.Device(0).name()
    except Exception:
        return "none"


def run_cell(
    *,
    backend: str,
    prg: str,
    depth: int,
    kernel: str = "sbox",
    key_mode: str = "variable",
    mode: str = "full",
    threads: int = 0,
    seed: Optional[bytes] = None,
    warmup: int = 5,
    repeats: int = 30,
) -> BenchResult:
    seed = seed if seed is not None else bytes(range(16))

    def expand_once() -> None:
        t = GGMTree(prg=prg, depth=depth, seed=seed, key_mode=key_mode)
        t.expand(backend=backend, kernel=kernel, mode=mode, threads=threads)

    for _ in range(warmup):
        expand_once()

    times: list[float] = []
    for _ in range(repeats):
        s = time.perf_counter()
        expand_once()
        times.append(time.perf_counter() - s)

    med = statistics.median(times)
    q75 = float(np.quantile(times, 0.75))
    q25 = float(np.quantile(times, 0.25))
    leaves = 1 << depth

    return BenchResult(
        backend=backend,
        prg=prg,
        kernel=kernel,
        key_mode=key_mode,
        threads=threads,
        depth=depth,
        mode=mode,
        median_seconds=med,
        iqr_seconds=q75 - q25,
        min_seconds=min(times),
        leaves_per_sec=leaves / med,
        bytes_per_sec=(leaves * 16) / med,
        repeats=repeats,
        warmup=warmup,
        seed_hex=seed.hex(),
        git_sha=_git_sha(),
        cpu_model=_cpu_model(),
        gpu_model=_gpu_model(),
    )


def save(res: BenchResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"{res.backend}-{res.prg}-{res.kernel}-{res.key_mode}"
        f"-t{res.threads}-m{res.mode}-d{res.depth:02d}.json"
    )
    path = out_dir / fname
    path.write_text(json.dumps(asdict(res), indent=2))
    return path
