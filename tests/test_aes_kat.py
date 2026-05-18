"""FIPS-197 AES-128 known-answer tests across CPU backends."""
import pytest

from ggm.kat import AES128_VECTORS
from ggm.ctypes_iface import has_aes_ni


@pytest.mark.parametrize("vec", AES128_VECTORS)
def test_aes_ref_matches_fips197(vec):
    from ggm.ctypes_iface import aes128_encrypt_block_ref
    assert aes128_encrypt_block_ref(vec.key, vec.plaintext) == vec.ciphertext


@pytest.mark.skipif(not has_aes_ni(), reason="CPU lacks AES-NI")
@pytest.mark.parametrize("vec", AES128_VECTORS)
def test_aes_ni_matches_fips197(vec):
    from ggm.ctypes_iface import aes128_encrypt_block_ni
    assert aes128_encrypt_block_ni(vec.key, vec.plaintext) == vec.ciphertext
