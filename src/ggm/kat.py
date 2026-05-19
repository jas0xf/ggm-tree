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


# Spongent-π[176] KAT — captured 2026-05-18 from spongent_ref.c (which uses the
# same constants as the GPU kernel; CPU and GPU produce bit-identical output).
# Cross-validation against the CHES 2011 paper's published π[176](0) vector is
# still TODO before final submission.
SPONGENT_PI176_ZERO = SpongentVector(
    input=bytes(22),
    output=bytes.fromhex("4b12cf7b99436d1dd133b61d6bf1a06b7d985c99a986"),
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

# GGM Spongent tree KAT at d=4 with seed = FIPS-197 C.1 key.
# Captured 2026-05-18 from expand_spongent_1t (Phase 2). 31 nodes; node 0 is
# the seed, nodes 15..30 are the leaves. CPU and GPU produce identical output.
GGM_SPONGENT_DEPTH4_KAT = {
    "seed": bytes.fromhex("000102030405060708090a0b0c0d0e0f"),
    "depth": 4,
    "tree_hex": [
        "000102030405060708090a0b0c0d0e0f",
        "7a3ff26dd940b47cbe2da502096d360c",
        "c6c6acc6f682612f63283f85fb74e4e0",
        "d31e9b2603b0b13653a3ab3f727f37d8",
        "0c0905812b2d8991151ec5751ac5c7a5",
        "25240104accb2db24713124cf6737e0d",
        "247f3aa709a8ef3065e100c3ae2e73f7",
        "d78870b76d137959a6a95a899647fee4",
        "14c1bacf00ee8e9631554b1876be4695",
        "67f0da9d4eff7424befc18f42682150c",
        "913cc34cf1572037daaf98f951239ffb",
        "37ce0ad2a949475fa2f15a28f7172c8d",
        "b8b55ea6219e8b578749275c4a55b061",
        "1378c6a52b96fa90401566f7085b2dc0",
        "681a1c632abcbeac78c555d34e86a239",
        "9c3c6bbcc3b80f18bf97c18ae19386ec",
        "838d8ea3b5667e9260bff3bbd3e80832",
        "432842d0f3416fed52e0b417868886ff",
        "8d665dc375e0e5971a123e5ef6561590",
        "03de8e9090574da647014e1d7ae16797",
        "097ab02be98fc0e460fdcfaa6b5d7638",
        "5149b3ebbab90d59254f479b69e59e80",
        "15823a1cb9d4779f641255dad1612828",
        "219ee45cf370fa5409976f6b5d0c61d7",
        "df2389ada0999f3f79ad4fe5f1726b37",
        "51a73aebdfc6c9d2fb08286deecee363",
        "5bdea172dac46bc3c4258edb8aa0425c",
        "adb8cc43d940d5364a98a350597f4f0f",
        "3810be91f4c4bbaaf339894fd5f4a137",
        "4e9da41d786f41dee9cac266b2e1e72d",
        "77cda772a18d9a3cc6ceeba337a4e5f1",
    ],
}
