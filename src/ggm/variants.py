"""Spongent variant table — constants for all supported permutation widths."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SpongentVariant:
    name: str
    width: int
    rounds: int
    lfsr_bits: int
    lfsr_init: int

    @property
    def nbytes(self) -> int:
        return self.width // 8


SPONGENT_VARIANTS: dict[str, SpongentVariant] = {
    "spongent-128": SpongentVariant(
        "spongent-128", width=136, rounds=70, lfsr_bits=7, lfsr_init=0x05
    ),
    "spongent-160": SpongentVariant(
        "spongent-160", width=176, rounds=80, lfsr_bits=7, lfsr_init=0x05
    ),
    "spongent-224": SpongentVariant(
        "spongent-224", width=240, rounds=100, lfsr_bits=7, lfsr_init=0x05
    ),
    "spongent-256": SpongentVariant(
        "spongent-256", width=272, rounds=140, lfsr_bits=8, lfsr_init=0x05
    ),
}

DEFAULT_SPONGENT = "spongent-160"


def get_variant(name: str) -> SpongentVariant:
    if name not in SPONGENT_VARIANTS:
        raise ValueError(
            f"Unknown Spongent variant {name!r}. "
            f"Choose from: {', '.join(SPONGENT_VARIANTS)}"
        )
    return SPONGENT_VARIANTS[name]
