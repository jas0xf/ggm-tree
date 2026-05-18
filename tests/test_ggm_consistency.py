"""Cross-backend equivalence and tree-KAT tests for GGM expansion."""

import pytest

from ggm.ctypes_iface import (
    expand_aes_sbox_1t,
    expand_aes_ni_1t,
    expand_aes_sbox_omp,
    has_aes_ni,
)
from ggm.kat import GGM_AES_DEPTH4_KAT


def test_aes_sbox_1t_depth_4_shape():
    seed = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    tree = expand_aes_sbox_1t(seed, depth=4)
    assert tree.shape == (31, 16)
    assert bytes(tree[0]) == seed


def test_aes_sbox_1t_matches_kat():
    kat = GGM_AES_DEPTH4_KAT
    tree = expand_aes_sbox_1t(kat["seed"], kat["depth"])
    expected = [bytes.fromhex(h) for h in kat["tree_hex"]]
    assert len(expected) == tree.shape[0]
    for i, exp in enumerate(expected):
        assert bytes(tree[i]) == exp, f"node {i} mismatch"


# Every other AES backend must agree byte-for-byte with the S-box reference.
SEED = bytes.fromhex("000102030405060708090a0b0c0d0e0f")


@pytest.mark.skipif(not has_aes_ni(), reason="CPU lacks AES-NI")
@pytest.mark.parametrize("depth", [4, 8, 12])
def test_aes_ni_matches_sbox(depth):
    sbox = expand_aes_sbox_1t(SEED, depth)
    ni = expand_aes_ni_1t(SEED, depth)
    assert (sbox == ni).all()


@pytest.mark.parametrize("depth,threads", [(4, 1), (8, 4), (12, 8)])
def test_aes_omp_matches_sbox(depth, threads):
    sbox = expand_aes_sbox_1t(SEED, depth)
    omp = expand_aes_sbox_omp(SEED, depth, threads)
    assert (sbox == omp).all()


# ---- GPU equivalence (requires CUDA device; auto-skipped on CPU-only boxes) ----


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [4, 8, 12, 16])
def test_aes_sbox_gpu_matches_cpu(depth):
    from ggm.host import gpu_expand_aes_sbox

    cpu = expand_aes_sbox_1t(SEED, depth)
    gpu = gpu_expand_aes_sbox(SEED, depth)
    assert (cpu == gpu).all()


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [4, 8, 12, 16])
def test_aes_ttable_gpu_matches_cpu(depth):
    from ggm.host import gpu_expand_aes_ttable

    cpu = expand_aes_sbox_1t(SEED, depth)
    gpu = gpu_expand_aes_ttable(SEED, depth)
    assert (cpu == gpu).all()


def _python_fixedkey_reference(seed: bytes, depth: int):
    """Python oracle for GPU fixed-key AES: G(s) = AES_K(s) || AES_K(s ⊕ 0x01) on byte 0."""
    from ggm.ctypes_iface import aes128_encrypt_block_ref
    import numpy as np

    FIXED_KEY = bytes.fromhex("00112233445566778899aabbccddeeff")
    nodes = [seed]
    internal = (1 << depth) - 1
    for i in range(internal):
        parent = nodes[i]
        in0 = parent
        in1 = bytes([parent[0] ^ 1]) + parent[1:]
        nodes.append(aes128_encrypt_block_ref(FIXED_KEY, in0))
        nodes.append(aes128_encrypt_block_ref(FIXED_KEY, in1))
    return np.array([list(n) for n in nodes], dtype=np.uint8)


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [4, 8, 12])
def test_aes_sbox_fixedkey_gpu_matches_python_reference(depth):
    from ggm.host import gpu_expand_aes_sbox_fixedkey

    ref = _python_fixedkey_reference(SEED, depth)
    gpu = gpu_expand_aes_sbox_fixedkey(SEED, depth)
    assert (ref == gpu).all()


@pytest.mark.gpu
@pytest.mark.parametrize("depth", [4, 8, 12])
def test_spongent_gpu_matches_cpu_when_ref_lands(depth):
    """Activates once US-2 lands the real π[176] permutation.

    For now (stub Spongent block = identity), CPU and GPU both produce
    'identity-tree' outputs, so they trivially match. The real test value
    of this case becomes meaningful when spongent_ref.c becomes correct.
    """
    from ggm.host import gpu_expand_spongent
    from ggm.ctypes_iface import expand_spongent_1t

    cpu = expand_spongent_1t(SEED, depth)
    gpu = gpu_expand_spongent(SEED, depth)
    assert (cpu == gpu).all()


@pytest.mark.gpu
@pytest.mark.parametrize("path_bits", ["0000", "1010", "11110000", "10101010"])
def test_aes_path_eval_matches_full_tree(path_bits):
    from ggm.host import gpu_expand_aes_sbox, gpu_path_eval_aes_sbox

    depth = len(path_bits)
    full = gpu_expand_aes_sbox(SEED, depth)
    idx = int(path_bits, 2)
    from_full = bytes(full[(1 << depth) - 1 + idx])
    from_eval = gpu_path_eval_aes_sbox(SEED, path_bits)
    assert from_full == from_eval
