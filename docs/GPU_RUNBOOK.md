# GPU Runbook — vast.ai RTX 3060

Step-by-step commands to run on the GPU box. Copy each block, paste into the
SSH session, paste the relevant output back here so we can iterate.

## 0. Prerequisites on the GPU box

```bash
# Connect (replace endpoint with the current vast.ai SSH command)
ssh -p 32630 root@120.238.149.205

# Confirm CUDA
nvidia-smi | head -15

# Install uv if not present
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Bring the repo over. Option A — push from local first, then clone:
#   (on local) git remote add origin git@github.com:<you>/ggm-tree
#   (on local) git push -u origin main
#   (on GPU)   git clone git@github.com:<you>/ggm-tree && cd ggm-tree
#
# Option B — direct rsync from your laptop into the GPU box:
#   (on local) rsync -avz --exclude .venv --exclude .pytest_cache \
#                 /home/jas0xf/Repo/Campus/UCSD/ECE268_HardwareSecurity/final_project/ \
#                 root@<host>:~/ggm-tree/
#   (on GPU)   cd ~/ggm-tree

# Install Python deps (CPU + GPU)
uv venv && uv sync --extra gpu --extra dev
source .venv/bin/activate

# Build the C library
make -C src/ggm/cpu

# Smoke test: 13 CPU tests should pass (same as local)
uv run pytest -q
```

**Paste back:** the `nvidia-smi` header and the pytest summary line.

## 1. PyCUDA smoke test

```bash
uv run python - <<'PY'
import pycuda.autoinit
import pycuda.driver as drv
d = drv.Device(0)
print(f"{d.name()} CC={d.compute_capability()} memory={d.total_memory()/2**30:.1f} GB")
print(f"SMs={d.get_attribute(drv.device_attribute.MULTIPROCESSOR_COUNT)}")
PY
```

**Paste back:** the four lines this prints.

## 2. US-3 — GPU AES S-box kernel (when I write it next session)

The plan at `docs/superpowers/plans/2026-05-18-ggm-tree-implementation.md` §Phase 3
spells out the kernel; I'll commit it next session. Once it lands:

```bash
git pull
uv run pytest -m gpu tests/test_ggm_consistency.py::test_aes_sbox_gpu_matches_cpu -v
```

**Paste back:** the full pytest output (PASS or full traceback).

## 3. US-4 — AES T-tables and bitsliced kernels

```bash
git pull
uv run pytest -m gpu tests/test_ggm_consistency.py::test_aes_ttable_gpu_matches_cpu -v
uv run pytest -m gpu tests/test_ggm_consistency.py::test_aes_bitslice_gpu_matches_cpu -v
```

## 4. US-5 — AES fixed-key

```bash
uv run pytest -m gpu tests/test_ggm_consistency.py::test_aes_sbox_fixedkey_gpu_matches_cpu -v
```

## 5. US-6 — Spongent GPU kernel

```bash
uv run pytest -m gpu tests/test_ggm_consistency.py::test_spongent_gpu_matches_cpu -v
```

## 6. US-9 — Full benchmark grid (GPU)

```bash
# Quick smoke first
uv run python -m bench.grid --filter d08 --depths 8

# Full grid (long — 30-60 min depending on bitsliced)
uv run python -m bench.grid --depths 4,6,8,10,12,14,16,18,20,22,24 --threads 1,4,8

# Render plots
uv run python -m bench.plot --speedup-depth 20
```

**Commit and push:**

```bash
git add bench/results/ report/figures/
git commit -m "bench: full GPU + CPU grid on RTX 3060"
git push
```

## 7. US-12.1 — Onboarding-from-scratch verification

On a brand-new vast.ai instance:

```bash
git clone <repo-url>
cd ggm-tree
# Follow only what README.md says, top to bottom
```

If anything in the README is unclear or broken, file an issue / open a PR.

## What I'll do remotely while you do the GPU runs

While you have the SSH session open, I can in parallel on the local box:

- Write CUDA kernel source (no compile-test possible locally without CUDA)
- Add more CPU experiments (different depths, OMP thread counts)
- Draft report sections that don't depend on GPU plots
- Run `lsp_diagnostics` and lint on the kernel code
- Build slides via `frontend-slides`

Just keep me posted on what you've pasted back and what worked.
