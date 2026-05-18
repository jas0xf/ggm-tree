"""Known-answer test vectors for AES-128 and Spongent-π[176]."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AESVector:
    key: bytes
    plaintext: bytes
    ciphertext: bytes


# FIPS-197 Appendix C.1 — AES-128 single-block test vector.
AES128_FIPS197_C1 = AESVector(
    key=bytes.fromhex("000102030405060708090a0b0c0d0e0f"),
    plaintext=bytes.fromhex("00112233445566778899aabbccddeeff"),
    ciphertext=bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a"),
)

AES128_VECTORS = [AES128_FIPS197_C1]


@dataclass(frozen=True)
class SpongentVector:
    input: bytes  # 22 bytes (176 bits)
    output: bytes  # 22 bytes after one π[176] application


# Spongent-π[176] KAT scaffold. The literal output is captured empirically by
# Phase 2 once spongent_ref.c is implemented, then cross-checked against the
# Spongent CHES 2011 paper / reference C code.
SPONGENT_PI176_ZERO = SpongentVector(
    input=bytes(22),
    output=bytes(22),  # populated in Phase 2 Task 2.3
)
SPONGENT_VECTORS = [SPONGENT_PI176_ZERO]


# GGM AES tree KAT at d=4 with seed = FIPS-197 C.1 key.
# Frozen 2026-05-18 from expand_aes_sbox_1t after FIPS-197 single-block KAT passed.
# 31 nodes total (2^5 - 1). Node 0 is the seed; nodes 15..30 are the leaves.
GGM_AES_DEPTH4_KAT = {
    "seed": bytes.fromhex("000102030405060708090a0b0c0d0e0f"),
    "depth": 4,
    "tree_hex": [
        "000102030405060708090a0b0c0d0e0f",
        "c6a13b37878f5b826f4f8162a1c8d879",
        "7346139595c0b41e497bbde365f42d0a",
        "2c578f7927a949d3b511ae8fb69145c6",
        "b75b1a66b8a4213ab3f5d73e3ba98a87",
        "cdbd38925be0ebd4eddb4aeabcd4ef6a",
        "0e6df65adcb33d311ea267e133067c0d",
        "7fd33c93316241be4be33fa21eb6641c",
        "66804fa3a13a7e391ca2cde37c7c9ecf",
        "b128c1c4cb3303a0076ee36d473058ab",
        "d20d33ddeab9d7f8215bd15dd7344cea",
        "932cedba9680d94041d7343ba85d97e0",
        "a264060c84ac851e1f58ee8b00cd55cb",
        "453031c983c66f999416fa25645e7a5c",
        "a75aba00fd2e01b67371b621f7c01dc3",
        "8384e6cd73588bb3ba120fb086fe4cfc",
        "53815c9870fabcdce3251ae9baa10ddd",
        "e71019b78881340cbf8e826c6ed63bc5",
        "8190d97a1edb7595225a77002d04e321",
        "7cfedda06e53b08af01895a789cd36ff",
        "b151c33f98011f330b0b2a94603f9880",
        "10e37c545d91e9d235a14588de4a9d3e",
        "6dd0cf97005133e4b84f299187465c36",
        "deffb0aef446e33c6d0be1aacb734df2",
        "81854efddeee7f59bfa8c806c3cbd445",
        "736db983a790531ad4e6a17dcb9ccb98",
        "3bb49186485518ef6f2170fa10eee8fb",
        "554b5ff9a26a57fdab69bc73aef64123",
        "34760866267ff6cbdb822d75459b5655",
        "0d7101e88ed03938b037a7db63cfd7fc",
        "4c605f3b89b0a3865acedb434ca39d3f",
    ],
}

# GGM Spongent tree KAT at d=4 (populated in Phase 2 Task 2.4).
GGM_SPONGENT_DEPTH4_KAT = {
    "seed": bytes.fromhex("000102030405060708090a0b0c0d0e0f"),
    "depth": 4,
    "tree_hex": [],
}
