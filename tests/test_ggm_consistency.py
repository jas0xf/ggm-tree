"""Cross-backend equivalence and tree-KAT tests for GGM expansion."""
import pytest

from ggm.ctypes_iface import expand_aes_sbox_1t
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
