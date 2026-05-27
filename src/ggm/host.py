"""PyCUDA kernel loaders and tree-expansion launchers.

Imported lazily by tree.py only when a GPU backend is requested, so this
module's import-time CUDA initialization does not fire on CPU-only machines.
"""

from __future__ import annotations
import functools
from pathlib import Path

import numpy as np

import pycuda.driver as drv
import pycuda.autoinit  # noqa: F401  -- side-effectful: establishes the context
from pycuda.compiler import SourceModule

_KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
_NVCC_OPTIONS = ["-O3", "-arch=sm_86", "--use_fast_math"]


@functools.lru_cache(maxsize=None)
def _load_module(
    src_filename: str, extra_defines: tuple[tuple[str, str], ...] = ()
) -> SourceModule:
    src = (_KERNEL_DIR / src_filename).read_text()
    opts = list(_NVCC_OPTIONS) + [f"-D{k}={v}" for k, v in extra_defines]
    return SourceModule(src, options=opts, no_extern_c=True)


# ---- AES constants (S-box + Rcon + T-tables) ----------------------------------------

_SBOX = bytes(
    [
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,
        0x04,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        0x05,
        0x9A,
        0x07,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,
        0x09,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,
        0x53,
        0xD1,
        0x00,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        0x02,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        0x06,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        0x08,
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        0x03,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,
    ]
)
_RCON = bytes([0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36])


def _mul2(a: int) -> int:
    return ((a << 1) ^ (((a >> 7) & 1) * 0x1B)) & 0xFF


def _mul3(a: int) -> int:
    return _mul2(a) ^ a


def _compute_t_tables() -> tuple[bytes, bytes, bytes, bytes]:
    t0 = bytearray()
    t1 = bytearray()
    t2 = bytearray()
    t3 = bytearray()
    for a in range(256):
        s = _SBOX[a]
        b2 = _mul2(s)
        b3 = _mul3(s)
        # Little-endian uint32 layout matches the kernel's pack_le32() usage.
        t0.extend([b2, s, s, b3])
        t1.extend([b3, b2, s, s])
        t2.extend([s, b3, b2, s])
        t3.extend([s, s, b3, b2])
    return bytes(t0), bytes(t1), bytes(t2), bytes(t3)


def _copy_constant(mod: SourceModule, name: str, data: bytes) -> None:
    ptr, _ = mod.get_global(name)
    drv.memcpy_htod(ptr, np.frombuffer(data, dtype=np.uint8))


def _upload_aes_constants(mod: SourceModule) -> None:
    _copy_constant(mod, "SBOX_C", _SBOX)
    _copy_constant(mod, "RCON_C", _RCON)


def _upload_t_tables(mod: SourceModule) -> None:
    for name, data in zip(("T0_C", "T1_C", "T2_C", "T3_C"), _compute_t_tables()):
        _copy_constant(mod, name, data)


# ---- Variable-key AES expansion -----------------------------------------------------


def _per_level_launch(fn, tree_gpu, depth: int) -> None:
    for level in range(depth):
        n = 1 << level
        block = min(256, max(1, n))
        grid = (n + block - 1) // block
        fn(tree_gpu, np.uint32(level), block=(block, 1, 1), grid=(grid, 1))


def _alloc_and_seed_tree(seed: bytes, depth: int) -> tuple[drv.DeviceAllocation, int]:
    total = (1 << (depth + 1)) - 1
    buf = drv.mem_alloc(total * 16)
    drv.memcpy_htod(buf, np.frombuffer(seed, dtype=np.uint8))
    return buf, total


def gpu_expand_aes_sbox(seed: bytes, depth: int) -> np.ndarray:
    mod = _load_module("aes_kernel.cu")
    _upload_aes_constants(mod)
    fn = mod.get_function("ggm_aes_sbox_expand_level")
    tree_gpu, total = _alloc_and_seed_tree(seed, depth)
    _per_level_launch(fn, tree_gpu, depth)
    out = np.empty(total * 16, dtype=np.uint8)
    drv.memcpy_dtoh(out, tree_gpu)
    tree_gpu.free()
    return out.reshape(-1, 16)


def gpu_expand_aes_ttable(seed: bytes, depth: int) -> np.ndarray:
    mod = _load_module("aes_kernel.cu")
    _upload_aes_constants(mod)
    _upload_t_tables(mod)
    fn = mod.get_function("ggm_aes_ttable_expand_level")
    tree_gpu, total = _alloc_and_seed_tree(seed, depth)
    _per_level_launch(fn, tree_gpu, depth)
    out = np.empty(total * 16, dtype=np.uint8)
    drv.memcpy_dtoh(out, tree_gpu)
    tree_gpu.free()
    return out.reshape(-1, 16)


def gpu_expand_aes_bitslice(seed: bytes, depth: int) -> np.ndarray:
    # Bitsliced kernel TBD (Phase 4 Task 4.3, time-boxed).
    # Until that lands, fall through to T-tables so the public API still works
    # and downstream benchmarks compare to a real run rather than crashing.
    return gpu_expand_aes_ttable(seed, depth)


# ---- Fixed-key AES expansion --------------------------------------------------------

_FIXED_KEY = bytes.fromhex("00112233445566778899aabbccddeeff")  # public, fixed test key


def _expand_key_python(key16: bytes) -> bytes:
    rk = bytearray(176)
    rk[:16] = key16
    for i in range(16, 176, 4):
        t = bytearray(rk[i - 4 : i])
        if i % 16 == 0:
            r = t[0]
            t[0] = _SBOX[t[1]] ^ _RCON[i // 16]
            t[1] = _SBOX[t[2]]
            t[2] = _SBOX[t[3]]
            t[3] = _SBOX[r]
        for k in range(4):
            rk[i + k] = rk[i - 16 + k] ^ t[k]
    return bytes(rk)


def gpu_expand_aes_sbox_fixedkey(seed: bytes, depth: int) -> np.ndarray:
    mod = _load_module("aes_kernel_fixedkey.cu")
    _copy_constant(mod, "SBOX_C", _SBOX)
    _copy_constant(mod, "FIXED_RK", _expand_key_python(_FIXED_KEY))
    fn = mod.get_function("ggm_aes_sbox_fixedkey_expand_level")
    tree_gpu, total = _alloc_and_seed_tree(seed, depth)
    _per_level_launch(fn, tree_gpu, depth)
    out = np.empty(total * 16, dtype=np.uint8)
    drv.memcpy_dtoh(out, tree_gpu)
    tree_gpu.free()
    return out.reshape(-1, 16)


# ---- Spongent expansion -------------------------------------------------------------

_SP_SBOX = bytes(
    [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]
)


def _compute_sp_player(width: int = 176) -> bytes:
    pmult = width // 4
    pmod = width - 1
    out = bytearray(width)
    for j in range(width):
        out[j] = j if j == pmod else (j * pmult) % pmod
    return bytes(out)


def _sp_defines(
    width: int, rounds: int, lfsr_bits: int, lfsr_init: int
) -> tuple[tuple[str, str], ...]:
    return (
        ("SP_WIDTH", str(width)),
        ("SP_BYTES", str(width // 8)),
        ("SP_ROUNDS", str(rounds)),
        ("SP_PMULT", str(width // 4)),
        ("SP_PMOD", str(width - 1)),
        ("SP_LFSR_BITS", str(lfsr_bits)),
        ("SP_LFSR_INIT", hex(lfsr_init)),
        ("SP_HI_BYTE", str((width - lfsr_bits) // 8)),
        ("SP_HI_SHIFT", str((width - lfsr_bits) % 8)),
    )


def gpu_expand_spongent(
    seed: bytes,
    depth: int,
    width: int = 176,
    rounds: int = 80,
    lfsr_bits: int = 7,
    lfsr_init: int = 0x05,
) -> np.ndarray:
    defines = _sp_defines(width, rounds, lfsr_bits, lfsr_init)
    mod = _load_module("spongent_kernel.cu", extra_defines=defines)
    _copy_constant(mod, "SP_SBOX", _SP_SBOX)
    _copy_constant(mod, "SP_PLAYER", _compute_sp_player(width))
    fn = mod.get_function("ggm_spongent_expand_level")
    tree_gpu, total = _alloc_and_seed_tree(seed, depth)
    _per_level_launch(fn, tree_gpu, depth)
    out = np.empty(total * 16, dtype=np.uint8)
    drv.memcpy_dtoh(out, tree_gpu)
    tree_gpu.free()
    return out.reshape(-1, 16)


# ---- Single-leaf path eval ----------------------------------------------------------


def gpu_path_eval_aes_sbox(seed: bytes, path_bits: str) -> bytes:
    """Single-leaf AES PRF evaluation via the path-eval kernel."""
    depth = len(path_bits)
    if depth == 0:
        return seed
    path_idx = int(path_bits, 2)
    mod = _load_module("aes_kernel.cu")
    _upload_aes_constants(mod)
    fn = mod.get_function("ggm_aes_sbox_path_eval")
    seed_gpu = drv.mem_alloc(16)
    leaf_gpu = drv.mem_alloc(16)
    drv.memcpy_htod(seed_gpu, np.frombuffer(seed, dtype=np.uint8))
    fn(
        seed_gpu,
        np.uint64(path_idx),
        np.uint32(depth),
        leaf_gpu,
        block=(1, 1, 1),
        grid=(1, 1),
    )
    out = np.empty(16, dtype=np.uint8)
    drv.memcpy_dtoh(out, leaf_gpu)
    seed_gpu.free()
    leaf_gpu.free()
    return bytes(out)
