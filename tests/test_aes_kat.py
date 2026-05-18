"""FIPS-197 AES-128 known-answer tests across CPU backends."""
import pytest

from ggm.kat import AES128_VECTORS


@pytest.mark.parametrize("vec", AES128_VECTORS)
def test_aes_ref_matches_fips197(vec):
    from ggm.ctypes_iface import aes128_encrypt_block_ref
    assert aes128_encrypt_block_ref(vec.key, vec.plaintext) == vec.ciphertext
