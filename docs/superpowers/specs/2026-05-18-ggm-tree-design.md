# GGM Tree on GPU — Design Spec

**Project:** UCSD ECE268 Hardware Security Final Project
**Topic:** GGM PRF tree from length-doubling PRG, with PRG-1 = AES-128 and PRG-2 = Spongent-π
**Date:** 2026-05-18
**Status:** Draft for user review

---

## 1. Goal and Scope

Build a GGM pseudorandom function tree on the GPU using PyCUDA, with two length-doubling PRGs (AES-128 and Spongent-π[176]) implemented from scratch as CUDA kernels. Support full tree expansion for depth d ∈ [4, 24] and varying 128-bit seeds. Benchmark every backend against multiple CPU baselines and analyze the performance–security trade-offs.

The deliverables for the course are:

1. A **10-minute presentation recording**.
2. A **2.5-minute in-class presentation**.
3. A **GitHub repository** with a fully operational CPU and GPU implementation and full group-member contribution history.
4. A **5-page IEEE double-column report** (background, design choices, implementation, results).

This spec is the contract for the technical work that feeds those deliverables.

## 2. Background

### 2.1 GGM construction

Let `G : {0,1}^n → {0,1}^{2n}` be a length-doubling PRG, split as `G(s) = (G_0(s), G_1(s))` with each half `n` bits. For a `d`-bit input `x = x_1 x_2 … x_d`, the GGM PRF is

```
F_K(x) = G_{x_d}( G_{x_{d-1}}( … G_{x_2}( G_{x_1}(K) ) … ) )
```

Equivalently, view `F_K` as the leaf reached by walking a complete binary tree of depth `d` from the root `K`, taking the `x_i`-th child at level `i`. **Full tree expansion** means materializing every internal node and every leaf — `2^{d+1} − 1` nodes total at `n = 128` bits each, so `(2^{d+1} − 1) × 16 B` of storage.

### 2.2 AES-128 as a PRG

AES-128 is a 128-bit block cipher with a 128-bit key. The PRF view treats `AES_K(·)` as a PRF over inputs in `{0,1}^{128}`. A length-doubling PRG is built from it as

```
G_AES(s)  =  AES_s(0) || AES_s(1)
G_0(s) = AES_s(0),  G_1(s) = AES_s(1)
```

where `s` is the parent's 128-bit value used as a fresh AES key. This is the canonical "GGM-from-PRF" reading; we also implement a **fixed-key counter-style** variant `G_AES(s) = AES_K(s‖0) || AES_K(s‖1)` for comparison.

### 2.3 Spongent-π as a PRG

Spongent (Bogdanov, Knežević, Leander, Toz, Varıcı, Verbauwhede, CHES 2011) is a lightweight hash function built from PRESENT-style permutations `π[b]` for various widths `b`. We use the permutation directly as a PRG:

```
G_Spongent(s) = high_128( π[176]( pad(s) ) ) || next_128( π[176]( pad(s) ) )
```

That is, one π[176] call produces ≥ 256 output bits, which we split into two 128-bit halves. We also implement a **two-call** variant `G(s) = π(s‖0)[0:128] || π(s‖1)[0:128]` for comparison.

### 2.4 GPU execution model in two sentences

CUDA on Ampere (RTX 3060, CC 8.6) runs warps of 32 threads in lockstep across 28 streaming multiprocessors. Memory is hierarchical (global, L2, L1 + shared, constant, registers); coalesced 128-byte global accesses and shared-memory broadcast/bank patterns dominate kernel performance.

## 3. Architecture Overview

### 3.1 Repository layout

```
ggm-tree/
├── README.md
├── pyproject.toml                 # uv-managed; pycuda, numpy, matplotlib, pytest
├── src/
│   ├── ggm/
│   │   ├── __init__.py
│   │   ├── tree.py                # host: build/eval API, backend dispatch
│   │   ├── kernels/
│   │   │   ├── aes_kernel.cu      # PRG-1 GPU kernels + tree expansion
│   │   │   └── spongent_kernel.cu # PRG-2 GPU kernels + tree expansion
│   │   ├── cpu/
│   │   │   ├── aes_ref.c          # AES S-box reference (1T)
│   │   │   ├── aes_ni.c           # AES-NI intrinsics
│   │   │   ├── aes_omp.c          # OpenMP wrapper around aes_ref
│   │   │   ├── spongent_ref.c     # Spongent π[176] reference (1T)
│   │   │   └── spongent_omp.c     # OpenMP wrapper around spongent_ref
│   │   ├── host.py                # PyCUDA SourceModule compile / launch
│   │   └── kat.py                 # KAT runner (FIPS-197, Spongent CHES vectors)
│   └── bench/
│       ├── bench_gpu.py
│       ├── bench_cpu.py
│       └── plot.py
├── tests/                         # pytest: KATs + cross-backend equivalence
├── notebooks/
│   └── demo.ipynb                 # live demo + plots
├── report/                        # IEEE LaTeX sources
│   ├── main.tex
│   ├── IEEEtran.cls
│   ├── figures/
│   └── refs.bib
├── slides/
│   ├── short/                     # 2.5-minute deck (frontend-slides)
│   └── long/                      # 10-minute deck (frontend-slides)
└── docs/superpowers/specs/        # this file lives here
```

### 3.2 Public Python API

```python
tree = GGMTree(prg="aes", depth=20, seed=K,        # prg ∈ {"aes", "spongent"}
               key_mode="variable")                 # AES only: {"variable", "fixed"}; Spongent ignores
tree.expand(backend="gpu",                          # backend ∈ {"gpu", "cpu_1t", "cpu_aesni", "cpu_omp"}
            mode="full",                            # mode ∈ {"full", "rolling", "leaves"}
            kernel="sbox",                          # kernel ∈ {"sbox", "ttable", "bitslice"} (AES only)
            spongent_calls=1)                       # Spongent only: {1, 2} (PRG construction)
leaf = tree.eval("01101010"*16, backend="gpu")     # single-leaf path eval
leaves = tree.leaves()                              # → np.ndarray shape (2**d, 16), uint8
```

`key_mode="fixed"` loads a separately compiled GPU module that uses `__constant__` round keys; `key_mode="variable"` uses the per-thread key-schedule module. Both are built at install time; selection is runtime.

## 4. Cryptographic Design

### 4.1 AES PRG kernels (three variants)

All three implement `G(s) = AES_s(0) || AES_s(1)` under variable-key. We also ship a parallel set of kernels (compiled into a separate PyCUDA module) for the fixed-key construction `G(s) = AES_K(s‖0) || AES_K(s‖1)` with the public `K` in `__constant__` memory and round keys precomputed once on the host. The host-side `key_mode` parameter selects which module is loaded.

| Kernel | S-box / round table location | Key material location | Notes |
|---|---|---|---|
| **v1: S-box** | `__constant__` (256 B) | registers (variable-key) or `__constant__` (fixed-key) | Simplest. Constant-memory broadcast cache works when all threads index the same byte; index-dependent access falls back to serialized lookups. Used as the always-correct baseline. |
| **v2: T-tables** | `__shared__` (4 KB = 4×256×4 B) | registers / `__constant__` | ~3× faster expected. Index-dependent shared-memory access is bank-pattern-dependent → **timing-variable**; report uses this to discuss cache/bank side channels. |
| **v3: Bitsliced** | none (logical S-box) | registers (32 keys per warp) | Constant-time by construction. 32 nodes per warp processed in parallel; complex implementation. Time-boxed; ship without it if not ready. |

For variable-key AES, each thread computes its own AES-128 key schedule (44 × `uint32_t`) and keeps it in registers; for fixed-key, the schedule is precomputed by the host and uploaded to constant memory once.

### 4.2 Spongent PRG kernel

**Variant:** Spongent-π[176] — 176-bit permutation, 80 rounds.

Per round (×80):
1. `state ^= lCounter`     — 7-bit LFSR round constant, XORed at fixed bit positions.
2. `state.sBoxLayer()`     — 4-bit PRESENT-style S-box applied to each of 44 nibbles.
3. `state.pLayer()`        — bit-permutation `state[i] → state[(i · 176/4) mod 175]` (non-MSB) plus an MSB fixed point.

Kernel layout:
- One thread = one parent node.
- State held in three `uint64_t` registers (the high 16 bits are unused).
- S-box: 16-byte lookup in `__constant__` memory, with an alternate bitsliced implementation for constant-time comparison.
- pLayer: 176-entry index table in `__constant__`, alternated against an unrolled bit-shuffle implementation.
- 80-round loop fully unrolled at compile time.

**Primary PRG construction:** one π[176] call per node (single-call mode). **Comparison construction:** two π calls with domain-separated inputs (two-call mode). Both shipped, both benchmarked.

### 4.3 PRG correctness verification

- **AES KAT:** FIPS-197 Appendix C single-block vectors; round-trip equality across `aes_ref.c`, `aes_ni.c`, `aes_kernel.cu` (all three GPU variants).
- **Spongent KAT:** original Spongent CHES 2011 paper test vectors; round-trip equality across `spongent_ref.c` and `spongent_kernel.cu`.
- **GGM tree KAT:** for fixed seed `K = 0x0123…EF` and `d = 4`, hard-code the expected 16 leaves (computed once with `aes_ref.c`) and assert bit-exact equality across every backend × kernel combination.

## 5. Memory Layout

### 5.1 Host

Pinned host buffer (`cuMemAllocHost`) of `(2^{d+1} − 1) × 16 B` for asynchronous D2H copy of the full tree. At `d = 24` that is 512 MB pinned, which we avoid by default: the host buffer caps at `d = 22` (128 MB pinned); for `d ∈ {23, 24}` we use the `mode="leaves"` path that keeps results on the GPU and copies only spot-check samples.

### 5.2 GPU — three tree modes

| Mode | Buffer | Footprint at d=24 | Use |
|---|---|---|---|
| `full` | flat global `nodes[0 : 2^{d+1}−1]` BFS array; parent `i`, children `2i+1, 2i+2` | 512 MB | Primary "full tree expansion" path; default for `d ≤ 22` |
| `rolling` | two ping-pong level buffers (current, next) | ~384 MB peak (leaf level 256 MB + previous level 128 MB during the final transition) | Memory–completeness trade-off plot; discards all non-leaf internals after each transition |
| `leaves` | leaf row only + sample paths | 256 MB (leaves) | Required for `d ∈ {23, 24}` |

### 5.3 GPU memory hierarchy

| Resource | Location | Size | Rationale |
|---|---|---|---|
| AES S-box (v1) | `__constant__` | 256 B | broadcast cache when index is uniform |
| AES T-tables (v2) | `__shared__` per block | 4 KB | per-block fast access, avoids constant-mem serialization on index-dependent reads |
| AES round keys (variable-key) | registers | 176 B/thread | each thread has its own schedule |
| AES round keys (fixed-key) | `__constant__` | 176 B | shared across all threads |
| Spongent state | registers | 24 B/thread | three `uint64_t` per node |
| Spongent S-box | `__constant__` or bitsliced | 16 B / 0 B | small; both variants implemented |
| Spongent pLayer table | `__constant__` | 176 B | indirect bit-permutation lookup |
| Tree storage | global | up to 512 MB | the actual output |
| Pinned host mirror | host pinned | up to 128 MB | async D2H |

### 5.4 Coalescing

At level `ℓ`, thread `t` reads parent at index `2^ℓ − 1 + t` (16 B) and writes two children at `2^{ℓ+1} − 1 + 2t` and `… + 2t + 1`. A warp of 32 threads reads `512 B` contiguous → 4 sectors of 128 B on Ampere — fully coalesced. Children writes are also contiguous within each child slot (two 16 B strides per thread = 32 B per child, contiguous across a warp = 1 KB contiguous = 8 sectors).

## 6. Parallelization Strategy

### 6.1 Per-level kernel launch (primary)

For `ℓ = 0 … d − 1`:

```
total_threads = 2^ℓ
block         = min(256, 2^ℓ)
grid          = ceil(total_threads / block)
kernel_expand<<<grid, block>>>(tree, ℓ)
```

Thread body:

```
i      = blockIdx.x * blockDim.x + threadIdx.x
parent = tree[(1 << ℓ) - 1 + i]
(c0, c1) = G(parent)
tree[(1 << (ℓ+1)) - 1 + 2*i]     = c0
tree[(1 << (ℓ+1)) - 1 + 2*i + 1] = c1
```

Pros: trivially correct, easy to reason about, exposes the "shallow levels underutilize the GPU" story explicitly. Cons: kernel launch overhead at shallow `ℓ`.

### 6.2 Persistent multi-level kernel (Approach B, follow-on)

One kernel covering depth range `(ℓ_lo, ℓ_hi)`: blocks cooperate via `__syncthreads()` at each level boundary, ramping up active threads as `ℓ` increases. Reduces launch overhead from `d` launches to ~`d / k` for some chunk size `k`.

### 6.3 Subtree-per-block (Approach B v2)

Once a level has at least `grid_size` nodes, each block owns a subtree of remaining depth `d − ℓ`. Block-local intermediates live in shared memory; only the leaf row is written to global. Reduces global-memory traffic by ~½ × `(d − ℓ)`.

### 6.4 Single-leaf path-eval kernel

For `tree.eval(x)`: one thread, `d` sequential PRG calls, no tree storage. Used for the lazy GGM evaluation benchmark and as a spot-check oracle for the full expansion.

### 6.5 Streams and overlap

A single CUDA stream per (seed, depth) benchmark cell — the per-level kernel launches are serialized within that stream. When the harness sweeps multiple seeds at the same depth, two streams are used to pipeline H2D-of-seed + kernel + D2H-of-leaves across consecutive seeds.

### 6.6 Determinism

Each thread writes a unique tree slot (`2i+1, 2i+2`) so there are no write races. Cross-backend equivalence is enforced bit-exactly by the KAT suite.

## 7. CPU Baselines

| Baseline | Build flags | Notes |
|---|---|---|
| **1T-C reference** | `-O3 -march=native -fno-tree-vectorize` | Same algorithmic structure as the GPU S-box kernel. Honest single-thread baseline. |
| **AES-NI** | `-O3 -march=native -maes` | Intel intrinsics `_mm_aeskeygenassist_si128`, `_mm_aesenc_si128`, `_mm_aesenclast_si128`. AES path only. |
| **OpenMP** | `-O3 -march=native -fopenmp` | `#pragma omp parallel for` at each level. Static schedule. Sweeps `OMP_NUM_THREADS ∈ {1, 2, 4, 8, 16}`. |

All three exposed via a single `libggmcpu.so` and loaded into Python with `ctypes`. One C entry point per backend (signatures as enumerated in Section 6 of the brainstorm — see `src/ggm/cpu/ggmcpu.h`).

## 8. Benchmark Plan

### 8.1 Measurement protocol

- 5 warmup runs + 30 timed runs per (backend, prg, kernel, mode, depth, seed) cell.
- Report median ± IQR.
- GPU timing via `cudaEvent_t` start/stop around kernel; H2D and D2H copies timed separately and as combined end-to-end.
- CPU timing via `clock_gettime(CLOCK_MONOTONIC_RAW)`.
- Throughput metrics: **leaves/sec** and **bytes/sec**.
- Each run is reproducible from a fixed seed schedule.
- Results stored as JSON under `bench/results/{backend}-{prg}-{kernel}-{depth}.json` with metadata (GPU model, CPU model, OMP threads, git SHA, kernel variant flags).

### 8.2 Plots (matplotlib, IEEE-styled, saved as PDF + PNG)

1. Throughput vs depth, one curve per backend (two figures: AES, Spongent).
2. GPU vs CPU speedup bar chart at `d = 20`.
3. AES kernel comparison: S-box vs T-tables vs bitsliced.
4. AES vs Spongent on GPU, same axes — illustrates the ~20× gap.
5. Memory-mode trade-off (`full` vs `rolling` vs `leaves`).
6. Per-level kernel-launch overhead vs depth.
7. (Optional) AES timing histogram from `RDTSC` samples — S-box vs AES-NI — for the constant-time discussion.

## 9. Testing

- `tests/test_aes_kat.py` — FIPS-197 vectors on every AES backend × kernel.
- `tests/test_spongent_kat.py` — Spongent CHES vectors on every Spongent backend.
- `tests/test_ggm_consistency.py` — bit-exact cross-backend equality at `d ∈ {4, 8, 12}`.
- `tests/test_tree_indexing.py` — BFS index math (`parent → children`, level boundaries).
- Edge cases: `d = 0`, `d = 1`, all-zero seed, all-one seed, NumPy-seeded random seed.

CI: GitHub Actions on `ubuntu-22.04`, CPU-only (`pytest`, `clang-format`, `black`); GPU benchmarks run manually on the vast.ai RTX 3060 box and committed as artifacts under `bench/results/`.

## 10. Report Plan (5-page IEEE)

| Section | Pages | Content |
|---|---|---|
| Abstract + Intro | 0.5 | Motivation, contributions |
| Background | 0.75 | GGM, AES-128, Spongent-π[176], CUDA execution model |
| Design Choices | 1.25 | AES var-key vs fixed-key, Spongent 1-call vs 2-call, kernel variants, memory modes, parallelism strategies |
| Implementation | 1.0 | Repo layout, PyCUDA glue, CPU baselines, GGM-tree-on-GPU mapping diagram |
| Results | 1.0 | Throughput, AES-vs-Spongent gap, GPU-vs-CPU speedup, memory-mode trade-off, AES timing-leak observation |
| Discussion / Future | 0.25 | Constant-time on GPU, Spongent hardware-vs-software gap |
| References | 0.25 | GGM '86, Spongent CHES '11, FIPS-197, NVIDIA Ampere whitepaper, course slides |

Class: `IEEEtran` with `\documentclass[10pt,conference]{IEEEtran}`. Figures: GGM tree diagram, GPU memory hierarchy mapping, plus the throughput plots from §8.

## 11. Presentations

### 11.1 2.5-minute in-class deck (3–4 slides)

1. Title + GGM in one diagram.
2. AES vs Spongent — block cipher vs sponge, hardware-supported vs hardware-targeted.
3. Headline GPU-vs-CPU speedup at `d = 20`.
4. Take-away.

### 11.2 10-minute recorded deck (10–12 slides)

1. Title, team, problem.
2. GGM construction (animated tree).
3. AES PRG construction + kernel variants.
4. Spongent PRG construction + kernel.
5. Memory layout (hierarchy table).
6. Parallelization strategy (per-level → persistent → subtree-block).
7. CPU baselines.
8. Results — throughput, speedup.
9. Trade-offs — constant-time, memory mode.
10. Conclusion + future work.

Both decks built with `frontend-slides:frontend-slides` (HTML/CSS); shared CSS and the same figures as the PDF report.

## 12. Repo & Group Workflow

- `main` (protected, PR-only), `dev` (integration), `feat/<member>-<topic>` (per-member feature branches).
- Squash-merge into `dev`; fast-forward into `main` at milestones.
- Each member opens PRs from their own GitHub account so the contribution graph reflects real participation.
- `CONTRIBUTORS.md` lists each member's primary and secondary areas of ownership.

Suggested split for a four-member group (final split TBD by the user). Smaller groups merge rows; larger groups split AES kernels by variant.

| Member | Primary | Secondary |
|---|---|---|
| A | AES kernels (S-box, T-tables) + AES KAT | Benchmark harness |
| B | Spongent kernel + Spongent KAT | Plots / figures |
| C | CPU baselines (1T, AES-NI, OpenMP) + integration | Report writing |
| D | Bitsliced AES + parallelism optimizations | Slides |

Group composition (member count and names) is the one open item in this spec; it will be resolved when the user confirms.

## 13. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| PyCUDA install pain on the vast.ai image | Document a single working `pip install pycuda` recipe with explicit CUDA path; pin versions in `pyproject.toml`. |
| RTX 3060 memory pressure at `d = 24` | Use `mode="leaves"` by default for `d ≥ 23`; cap `mode="full"` at `d = 22`. |
| AES-NI not available on the vast.ai CPU | Runtime `cpuid` check; emit a clear warning and skip that single bar in the speedup plot if missing. |
| Bitsliced AES implementation overruns its time-box | Cap at one week; ship without v3 if not ready — the report still works with v1 + v2. |
| Spongent reference availability | Use the CHES 2011 paper test vectors and an independent Python implementation as a cross-check oracle. |
| Group workload imbalance | Public task board (`TASKS.md` plus GitHub Issues); weekly check-ins. |

## 14. Out of Scope

- Production-grade cryptographic library packaging.
- Side-channel resistance proofs beyond a qualitative discussion of constant-time on the GPU.
- AES-256 or other AES variants (locked to AES-128 for the PRG).
- Spongent variants other than π[176] in the primary path (π[160] and π[256] are stretch goals).

## 15. Definition of Done

- All KATs pass on every backend × kernel combination.
- Full benchmark grid (CPU 1T, CPU AES-NI, CPU OMP {1,2,4,8,16}, GPU S-box, GPU T-tables, GPU bitsliced if shipped) × (AES, Spongent) × (d ∈ {4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}) collected and committed.
- All seven plots produced as PDF + PNG.
- 5-page IEEE report compiled and committed.
- 2.5-minute and 10-minute HTML decks rendered.
- 10-minute presentation recorded and uploaded.
- README onboarding works end-to-end on a fresh vast.ai instance.
- CONTRIBUTORS.md is accurate and each group member has merged PRs.
