"""ctypes binding to libggmcpu.so."""

from __future__ import annotations
import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parent / "cpu" / "libggmcpu.so"
_lib = ctypes.CDLL(str(_LIB_PATH))

# ---- AES single-block primitives ----------------------------------------------------
_lib.ggm_aes128_encrypt_block_ref.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
]
_lib.ggm_aes128_encrypt_block_ref.restype = None


def aes128_encrypt_block_ref(key: bytes, plaintext: bytes) -> bytes:
    assert len(key) == 16 and len(plaintext) == 16
    out = ctypes.create_string_buffer(16)
    _lib.ggm_aes128_encrypt_block_ref(key, plaintext, out)
    return bytes(out.raw[:16])


# ---- Spongent single-block primitive ------------------------------------------------
_lib.ggm_spongent_pi176_block_ref.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.ggm_spongent_pi176_block_ref.restype = None

_lib.ggm_spongent_block_generic.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint8,
    ctypes.c_char_p,
    ctypes.c_char_p,
]
_lib.ggm_spongent_block_generic.restype = None


def spongent_pi176_block(in_22: bytes) -> bytes:
    assert len(in_22) == 22
    out = ctypes.create_string_buffer(22)
    _lib.ggm_spongent_pi176_block_ref(in_22, out)
    return bytes(out.raw[:22])


def spongent_block_generic(
    width: int, rounds: int, lfsr_bits: int, lfsr_init: int, data: bytes
) -> bytes:
    nbytes = width // 8
    assert len(data) == nbytes
    out = ctypes.create_string_buffer(nbytes)
    _lib.ggm_spongent_block_generic(
        ctypes.c_int(width),
        ctypes.c_int(rounds),
        ctypes.c_int(lfsr_bits),
        ctypes.c_uint8(lfsr_init),
        data,
        out,
    )
    return bytes(out.raw[:nbytes])


# ---- Tree expansion -----------------------------------------------------------------
def _alloc_tree(depth: int) -> tuple[ctypes.Array[ctypes.c_ubyte], int]:
    total = ((1 << (depth + 1)) - 1) * 16
    return (ctypes.c_ubyte * total)(), total


_lib.ggm_expand_aes_sbox_1t.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint32,
    ctypes.c_char_p,
]
_lib.ggm_expand_aes_sbox_1t.restype = None


def expand_aes_sbox_1t(seed: bytes, depth: int) -> np.ndarray:
    assert len(seed) == 16 and 0 <= depth <= 24
    buf, _ = _alloc_tree(depth)
    _lib.ggm_expand_aes_sbox_1t(
        seed, ctypes.c_uint32(depth), ctypes.cast(buf, ctypes.c_char_p)
    )
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 16).copy()


_lib.ggm_expand_spongent_1t.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint32,
    ctypes.c_char_p,
]
_lib.ggm_expand_spongent_1t.restype = None

_lib.ggm_expand_spongent_generic_1t.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint8,
    ctypes.c_char_p,
    ctypes.c_uint32,
    ctypes.c_char_p,
]
_lib.ggm_expand_spongent_generic_1t.restype = None

_lib.ggm_expand_spongent_generic_omp.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint8,
    ctypes.c_char_p,
    ctypes.c_uint32,
    ctypes.c_char_p,
    ctypes.c_int,
]
_lib.ggm_expand_spongent_generic_omp.restype = None


def expand_spongent_1t(seed: bytes, depth: int) -> np.ndarray:
    assert len(seed) == 16 and 0 <= depth <= 24
    buf, _ = _alloc_tree(depth)
    _lib.ggm_expand_spongent_1t(
        seed, ctypes.c_uint32(depth), ctypes.cast(buf, ctypes.c_char_p)
    )
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 16).copy()


def expand_spongent_generic_1t(
    width: int,
    rounds: int,
    lfsr_bits: int,
    lfsr_init: int,
    seed: bytes,
    depth: int,
) -> np.ndarray:
    assert len(seed) == 16 and 0 <= depth <= 24
    buf, _ = _alloc_tree(depth)
    _lib.ggm_expand_spongent_generic_1t(
        ctypes.c_int(width),
        ctypes.c_int(rounds),
        ctypes.c_int(lfsr_bits),
        ctypes.c_uint8(lfsr_init),
        seed,
        ctypes.c_uint32(depth),
        ctypes.cast(buf, ctypes.c_char_p),
    )
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 16).copy()


def expand_spongent_generic_omp(
    width: int,
    rounds: int,
    lfsr_bits: int,
    lfsr_init: int,
    seed: bytes,
    depth: int,
    threads: int = 0,
) -> np.ndarray:
    assert len(seed) == 16 and 0 <= depth <= 24
    buf, _ = _alloc_tree(depth)
    _lib.ggm_expand_spongent_generic_omp(
        ctypes.c_int(width),
        ctypes.c_int(rounds),
        ctypes.c_int(lfsr_bits),
        ctypes.c_uint8(lfsr_init),
        seed,
        ctypes.c_uint32(depth),
        ctypes.cast(buf, ctypes.c_char_p),
        ctypes.c_int(threads),
    )
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 16).copy()


# ---- AES-NI / OpenMP variants (stubs until Phase 7 lands) ---------------------------
_lib.ggm_aes128_encrypt_block_ni.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
]
_lib.ggm_aes128_encrypt_block_ni.restype = None


def aes128_encrypt_block_ni(key: bytes, plaintext: bytes) -> bytes:
    assert len(key) == 16 and len(plaintext) == 16
    out = ctypes.create_string_buffer(16)
    _lib.ggm_aes128_encrypt_block_ni(key, plaintext, out)
    return bytes(out.raw[:16])


_lib.ggm_expand_aes_ni_1t.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p]
_lib.ggm_expand_aes_ni_1t.restype = None


def expand_aes_ni_1t(seed: bytes, depth: int) -> np.ndarray:
    buf, _ = _alloc_tree(depth)
    _lib.ggm_expand_aes_ni_1t(
        seed, ctypes.c_uint32(depth), ctypes.cast(buf, ctypes.c_char_p)
    )
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 16).copy()


_lib.ggm_expand_aes_sbox_omp.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint32,
    ctypes.c_char_p,
    ctypes.c_int,
]
_lib.ggm_expand_aes_sbox_omp.restype = None


def expand_aes_sbox_omp(seed: bytes, depth: int, threads: int = 0) -> np.ndarray:
    buf, _ = _alloc_tree(depth)
    _lib.ggm_expand_aes_sbox_omp(
        seed,
        ctypes.c_uint32(depth),
        ctypes.cast(buf, ctypes.c_char_p),
        ctypes.c_int(threads),
    )
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 16).copy()


_lib.ggm_expand_spongent_omp.argtypes = [
    ctypes.c_char_p,
    ctypes.c_uint32,
    ctypes.c_char_p,
    ctypes.c_int,
]
_lib.ggm_expand_spongent_omp.restype = None


def expand_spongent_omp(seed: bytes, depth: int, threads: int = 0) -> np.ndarray:
    buf, _ = _alloc_tree(depth)
    _lib.ggm_expand_spongent_omp(
        seed,
        ctypes.c_uint32(depth),
        ctypes.cast(buf, ctypes.c_char_p),
        ctypes.c_int(threads),
    )
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, 16).copy()


def has_aes_ni() -> bool:
    """Detect Intel AES-NI on x86 by reading /proc/cpuinfo flags."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return "aes" in line.split(":", 1)[-1].split()
    except FileNotFoundError:
        pass
    return False
