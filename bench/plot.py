"""Render benchmark plots from bench/results/*.json into report/figures/."""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "figure.figsize": (3.4, 2.3),
        "savefig.bbox": "tight",
    }
)


def load_results(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def _label(d: dict) -> str:
    parts = [d["backend"]]
    if d["backend"] == "cpu_omp":
        parts.append(f"t={d['threads']}")
    if d.get("kernel", "sbox") != "sbox":
        parts.append(d["kernel"])
    if d.get("key_mode", "variable") == "fixed":
        parts.append("fixed-key")
    return "/".join(parts)


def plot_throughput_vs_depth(data: list[dict], prg: str, outfile: Path) -> None:
    fig, ax = plt.subplots()
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for d in data:
        if d["prg"] != prg:
            continue
        series[_label(d)].append((d["depth"], d["leaves_per_sec"]))
    for label, pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", linewidth=1.0, markersize=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("tree depth $d$")
    ax.set_ylabel("leaves / s")
    ax.set_title(f"{prg.upper()} GGM tree throughput")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile)
    fig.savefig(outfile.with_suffix(".png"), dpi=200)
    plt.close(fig)


def plot_speedup_bars(data: list[dict], depth: int, outfile: Path) -> None:
    bars = [
        (_label(d), d["leaves_per_sec"])
        for d in data
        if d["prg"] == "aes" and d["depth"] == depth
    ]
    if not bars:
        return
    bars.sort(key=lambda x: x[1])
    labels, vals = zip(*bars)
    fig, ax = plt.subplots(figsize=(3.4, 0.3 * len(labels) + 1.0))
    ax.barh(labels, vals)
    ax.set_xscale("log")
    ax.set_xlabel("leaves / s (log)")
    ax.set_title(f"AES @ d={depth}")
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile)
    fig.savefig(outfile.with_suffix(".png"), dpi=200)
    plt.close(fig)


def plot_aes_kernel_comparison(data, outfile: Path) -> None:
    fig, ax = plt.subplots()
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for d in data:
        if d["backend"] != "gpu" or d["prg"] != "aes":
            continue
        key = d["kernel"]
        if d.get("key_mode", "variable") == "fixed":
            key = f"{key} (fixed-key)"
        series[key].append((d["depth"], d["leaves_per_sec"]))
    if not series:
        return
    for label, pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", linewidth=1.0, markersize=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("tree depth $d$")
    ax.set_ylabel("leaves / s")
    ax.set_title("GPU AES kernel variants")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile)
    fig.savefig(outfile.with_suffix(".png"), dpi=200)
    plt.close(fig)


def plot_aes_vs_spongent_gpu(data, outfile: Path) -> None:
    fig, ax = plt.subplots()
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for d in data:
        if d["backend"] != "gpu":
            continue
        if d["prg"] == "aes" and (
            d["kernel"] != "sbox" or d.get("key_mode") == "fixed"
        ):
            continue
        series[d["prg"].upper()].append((d["depth"], d["leaves_per_sec"]))
    if not series:
        return
    for label, pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", linewidth=1.0, markersize=3, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("tree depth $d$")
    ax.set_ylabel("leaves / s")
    ax.set_title("AES vs Spongent on GPU (S-box, var-key)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile)
    fig.savefig(outfile.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results", type=Path)
    ap.add_argument("--figures", default="report/figures", type=Path)
    ap.add_argument("--speedup-depth", default=20, type=int)
    args = ap.parse_args(argv)

    data = load_results(args.results)
    if not data:
        print(f"[plot] no results in {args.results}")
        return 0

    plot_throughput_vs_depth(data, "aes", args.figures / "throughput_aes.pdf")
    plot_throughput_vs_depth(data, "spongent", args.figures / "throughput_spongent.pdf")
    plot_speedup_bars(
        data,
        args.speedup_depth,
        args.figures / f"speedup_d{args.speedup_depth:02d}.pdf",
    )
    plot_aes_kernel_comparison(data, args.figures / "aes_kernels.pdf")
    plot_aes_vs_spongent_gpu(data, args.figures / "aes_vs_spongent.pdf")
    print(f"[plot] wrote figures to {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
