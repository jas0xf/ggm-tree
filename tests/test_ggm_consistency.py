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
    ni   = expand_aes_ni_1t(SEED, depth)
    assert (sbox == ni).all()


@pytest.mark.parametrize("depth,threads", [(4, 1), (8, 4), (12, 8)])
def test_aes_omp_matches_sbox(depth, threads):
    sbox = expand_aes_sbox_1t(SEED, depth)
    omp  = expand_aes_sbox_omp(SEED, depth, threads)
    assert (sbox == omp).all()
