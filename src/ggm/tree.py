"""Public GGMTree API: backend dispatch + tree-array helpers.

Importing this module does NOT require CUDA; GPU backends are loaded lazily
on first GPU call and raise a clear error if PyCUDA / a CUDA device is missing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import ctypes_iface
from .variants import SPONGENT_VARIANTS, DEFAULT_SPONGENT, get_variant

_CPU_BACKENDS = {"cpu_1t", "cpu_aesni", "cpu_omp"}
_GPU_BACKENDS = {"gpu"}
_ALL_BACKENDS = _CPU_BACKENDS | _GPU_BACKENDS

_PRGS = {"aes", "spongent"}
_AES_KERNELS = {"sbox", "ttable", "bitslice"}
_KEY_MODES = {"variable", "fixed"}
_MODES = {"full", "rolling", "leaves"}


@dataclass
class GGMTree:
    prg: str
    depth: int
    seed: bytes
    key_mode: str = "variable"
    spongent_variant: str = DEFAULT_SPONGENT

    _tree: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.prg not in _PRGS:
            raise ValueError(f"prg must be one of {_PRGS}; got {self.prg!r}")
        if not isinstance(self.seed, (bytes, bytearray)) or len(self.seed) != 16:
            raise ValueError("seed must be 16 bytes")
        if not (0 <= self.depth <= 24):
            raise ValueError("depth must be in [0, 24]")
        if self.key_mode not in _KEY_MODES:
            raise ValueError(
                f"key_mode must be one of {_KEY_MODES}; got {self.key_mode!r}"
            )
        if self.prg == "spongent" and self.spongent_variant not in SPONGENT_VARIANTS:
            raise ValueError(
                f"spongent_variant must be one of {list(SPONGENT_VARIANTS)}; "
                f"got {self.spongent_variant!r}"
            )
        self.seed = bytes(self.seed)

    def expand(
        self,
        backend: str = "cpu_1t",
        kernel: str = "sbox",
        mode: str = "full",
        threads: int = 0,
    ) -> np.ndarray:
        """Materialize the tree using the chosen backend. Returns (N, 16) uint8 ndarray."""
        if backend not in _ALL_BACKENDS:
            raise ValueError(f"backend must be one of {_ALL_BACKENDS}; got {backend!r}")
        if kernel not in _AES_KERNELS:
            raise ValueError(f"kernel must be one of {_AES_KERNELS}; got {kernel!r}")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}; got {mode!r}")

        if backend in _GPU_BACKENDS:
            self._tree = self._expand_gpu(kernel=kernel, mode=mode)
        else:
            self._tree = self._expand_cpu(backend=backend, threads=threads)
        return self._tree

    def _expand_cpu(self, backend: str, threads: int) -> np.ndarray:
        if self.prg == "aes":
            if backend == "cpu_1t":
                return ctypes_iface.expand_aes_sbox_1t(self.seed, self.depth)
            if backend == "cpu_aesni":
                if not ctypes_iface.has_aes_ni():
                    raise RuntimeError("AES-NI not available on this CPU")
                return ctypes_iface.expand_aes_ni_1t(self.seed, self.depth)
            if backend == "cpu_omp":
                return ctypes_iface.expand_aes_sbox_omp(self.seed, self.depth, threads)
        elif self.prg == "spongent":
            v = get_variant(self.spongent_variant)
            if backend == "cpu_1t":
                return ctypes_iface.expand_spongent_generic_1t(
                    v.width,
                    v.rounds,
                    v.lfsr_bits,
                    v.lfsr_init,
                    self.seed,
                    self.depth,
                )
            if backend == "cpu_omp":
                return ctypes_iface.expand_spongent_generic_omp(
                    v.width,
                    v.rounds,
                    v.lfsr_bits,
                    v.lfsr_init,
                    self.seed,
                    self.depth,
                    threads,
                )
            if backend == "cpu_aesni":
                raise ValueError("AES-NI does not apply to Spongent")
        raise ValueError(
            f"unsupported combination prg={self.prg!r} backend={backend!r}"
        )

    def _expand_gpu(self, kernel: str, mode: str) -> np.ndarray:
        try:
            from . import host  # local import: pulls in PyCUDA
        except Exception as e:  # pragma: no cover - exercised only on no-CUDA machines
            raise RuntimeError(
                "GPU backend requires PyCUDA + a CUDA device; "
                "install the [gpu] extra and run on a CUDA-capable box"
            ) from e
        if self.prg == "aes":
            if self.key_mode == "variable" and kernel == "sbox":
                return host.gpu_expand_aes_sbox(self.seed, self.depth)
            if self.key_mode == "variable" and kernel == "ttable":
                return host.gpu_expand_aes_ttable(self.seed, self.depth)
            if self.key_mode == "variable" and kernel == "bitslice":
                return host.gpu_expand_aes_bitslice(self.seed, self.depth)
            if self.key_mode == "fixed" and kernel == "sbox":
                return host.gpu_expand_aes_sbox_fixedkey(self.seed, self.depth)
        elif self.prg == "spongent":
            return host.gpu_expand_spongent(self.seed, self.depth)
        raise ValueError(
            f"unsupported GPU combo prg={self.prg!r} kernel={kernel!r} key_mode={self.key_mode!r}"
        )

    def leaves(self) -> np.ndarray:
        """Return only the leaf row (2^depth × 16)."""
        if self._tree is None:
            raise RuntimeError("call expand() first")
        leaf_offset = (1 << self.depth) - 1
        # Full mode stores 2^(d+1)-1 nodes; rolling/leaves modes may store only 2^d.
        if self._tree.shape[0] == (1 << self.depth):
            return self._tree
        return self._tree[leaf_offset:]

    def eval(self, path_bits: str) -> bytes:
        """Single-leaf path-eval. Requires expand() first (uses materialized tree)."""
        if self._tree is None:
            raise RuntimeError(
                "call expand() first; GPU path-eval kernel lands in Phase 8"
            )
        if len(path_bits) != self.depth or set(path_bits) - {"0", "1"}:
            raise ValueError(f"path_bits must be a {self.depth}-char 0/1 string")
        idx = int(path_bits, 2)
        return bytes(self.leaves()[idx])
