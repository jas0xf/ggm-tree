# GGM Tree on GPU

UCSD ECE268 Hardware Security final project. Builds a GGM PRF tree on the GPU using
AES-128 and Spongent-π[176] as length-doubling PRGs. CPU baselines for comparison.

## Quick start

```bash
uv venv && uv sync --all-extras
source .venv/bin/activate
make -C src/ggm/cpu                          # builds libggmcpu.so
pytest                                       # CPU-only tests
pytest -m gpu                                # GPU tests (needs CUDA)
python -m bench.grid --filter d04            # smoke benchmark
python -m bench.grid                         # full grid (long; run on GPU box)
python -m bench.plot                         # render all plots
```

## Project layout

- `src/ggm/` — Python package (host glue, ctypes interface, public `GGMTree` API).
- `src/ggm/kernels/*.cu` — hand-written CUDA kernels (AES variable/fixed key, Spongent π[176]).
- `src/ggm/cpu/*.c` — CPU reference, AES-NI, OpenMP baselines; built into `libggmcpu.so`.
- `bench/` — measurement harness, grid driver, plotting.
- `tests/` — pytest suite (KATs, cross-backend equivalence, tree indexing).
- `report/` — IEEE 5-page LaTeX report.
- `slides/` — short (2.5 min) and long (10 min) HTML decks.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — design spec + implementation plan.

## Design and plan

- Design spec: `docs/superpowers/specs/2026-05-18-ggm-tree-design.md`
- Implementation plan: `docs/superpowers/plans/2026-05-18-ggm-tree-implementation.md`
