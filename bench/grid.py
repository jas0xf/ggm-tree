"""Drive the full backend × prg × kernel × depth benchmark grid.

CPU-only by default; GPU cells are skipped automatically if no CUDA device.
Filter via --filter to run a subset (e.g. --filter aes-sbox or --filter d08).
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

from bench.runner import run_cell, save


def _gpu_present() -> bool:
    try:
        import pycuda.autoinit  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _cells_cpu(depths: list[int], omp_threads: list[int]) -> Iterable[dict]:
    for d in depths:
        yield dict(backend="cpu_1t", prg="aes", kernel="sbox", depth=d)
        yield dict(backend="cpu_aesni", prg="aes", kernel="sbox", depth=d)
        for t in omp_threads:
            yield dict(backend="cpu_omp", prg="aes", kernel="sbox", depth=d, threads=t)


def _cells_gpu(depths: list[int]) -> Iterable[dict]:
    for d in depths:
        for kernel in ("sbox", "ttable", "bitslice"):
            yield dict(backend="gpu", prg="aes", kernel=kernel, depth=d)
        yield dict(backend="gpu", prg="aes", kernel="sbox", key_mode="fixed", depth=d)
        yield dict(backend="gpu", prg="spongent", kernel="sbox", depth=d)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GGM benchmark grid driver")
    ap.add_argument("--out", default="bench/results", type=Path)
    ap.add_argument("--filter", default="", help="substring filter on the cell tag")
    ap.add_argument("--depths", default="4,6,8,10,12,14",
                    help="comma-separated tree depths (default: 4..14 for fast CPU runs)")
    ap.add_argument("--threads", default="1,4,8",
                    help="comma-separated OpenMP thread counts")
    ap.add_argument("--skip-gpu", action="store_true",
                    help="never run GPU cells, even if a CUDA device is present")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    depths = [int(x) for x in args.depths.split(",")]
    threads = [int(x) for x in args.threads.split(",")]
    gpu = (not args.skip_gpu) and _gpu_present()

    cells = list(_cells_cpu(depths, threads))
    if gpu:
        cells += list(_cells_gpu(depths))

    print(f"[grid] {len(cells)} cells (gpu={'yes' if gpu else 'no'})")
    t0 = time.perf_counter()
    n_run = 0
    for cell in cells:
        tag = (
            f"{cell['backend']}-{cell['prg']}-{cell.get('kernel', 'sbox')}"
            f"-{cell.get('key_mode', 'variable')}"
            f"-t{cell.get('threads', 0)}-d{cell['depth']:02d}"
        )
        if args.filter and args.filter not in tag:
            continue
        print(f"[run] {tag}", flush=True)
        if args.dry_run:
            continue
        try:
            r = run_cell(**cell)
            save(r, args.out)
            n_run += 1
        except Exception as e:
            print(f"  [error] {e}", file=sys.stderr)
    print(f"[grid] {n_run} cells completed in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
