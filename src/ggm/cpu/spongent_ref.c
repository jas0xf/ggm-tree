/* src/ggm/cpu/spongent_ref.c — Generic Spongent permutation + GGM tree expansion.
 *
 * Supports all four Spongent variants with width >= 128 bits:
 *   Spongent-128 (π[136], 70 rounds, 17 bytes)
 *   Spongent-160 (π[176], 80 rounds, 22 bytes)
 *   Spongent-224 (π[240], 100 rounds, 30 bytes)
 *   Spongent-256 (π[272], 140 rounds, 34 bytes)
 *
 * The generic function takes (width, rounds, lfsr_bits, lfsr_init) as params.
 * The old ggm_spongent_pi176_block_ref / ggm_expand_spongent_1t are thin
 * wrappers for backward compatibility with existing tests.
 */
#include "ggmcpu.h"
#include <string.h>

#define SP_MAX_BYTES 34  /* 272 bits / 8 */

static const uint8_t SP4_SBOX[16] = {
    0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD,
    0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2,
};

static inline int sp_bit(const uint8_t *s, int i) {
    return (s[i >> 3] >> (i & 7)) & 1;
}

static inline void sp_setbit(uint8_t *s, int i, int v) {
    int byte = i >> 3, bit = i & 7;
    s[byte] = (uint8_t)((s[byte] & ~(1u << bit)) | ((v & 1) << bit));
}

static inline uint8_t sp_lfsr_step(uint8_t lc, int lfsr_bits) {
    uint8_t bit = ((lc >> (lfsr_bits - 1)) ^ (lc >> (lfsr_bits - 2))) & 1;
    return (uint8_t)(((lc << 1) | bit) & ((1u << lfsr_bits) - 1));
}

static void sp_round_generic(uint8_t *state, int width_bits, int lfsr_bits, uint8_t lc) {
    int nbytes = width_bits / 8;

    /* AddCounter: lc into low lfsr_bits, reversed lc into top lfsr_bits. */
    state[0] ^= lc;
    uint8_t rlc = 0;
    for (int k = 0; k < lfsr_bits; k++) {
        if ((lc >> k) & 1) rlc = (uint8_t)(rlc | (1u << (lfsr_bits - 1 - k)));
    }
    int hi_byte  = (width_bits - lfsr_bits) / 8;
    int hi_shift = (width_bits - lfsr_bits) % 8;
    state[hi_byte] ^= (uint8_t)(rlc << hi_shift);

    /* S-box layer per nibble. */
    for (int i = 0; i < nbytes; i++) {
        uint8_t lo = state[i] & 0xF;
        uint8_t hi = (state[i] >> 4) & 0xF;
        state[i] = (uint8_t)((SP4_SBOX[hi] << 4) | SP4_SBOX[lo]);
    }

    /* pLayer: j → (j * (width/4)) mod (width - 1), with j = width-1 fixed. */
    int player_mult = width_bits / 4;
    int player_mod  = width_bits - 1;
    uint8_t buf[SP_MAX_BYTES] = {0};
    for (int j = 0; j < width_bits; j++) {
        int dest = (j == player_mod) ? j : (j * player_mult) % player_mod;
        sp_setbit(buf, dest, sp_bit(state, j));
    }
    memcpy(state, buf, nbytes);
}

void ggm_spongent_block_generic(
    int width_bits, int rounds, int lfsr_bits, uint8_t lfsr_init,
    const uint8_t *in, uint8_t *out)
{
    int nbytes = width_bits / 8;
    uint8_t state[SP_MAX_BYTES];
    memcpy(state, in, nbytes);
    uint8_t lc = lfsr_init;
    for (int r = 0; r < rounds; r++) {
        sp_round_generic(state, width_bits, lfsr_bits, lc);
        lc = sp_lfsr_step(lc, lfsr_bits);
    }
    memcpy(out, state, nbytes);
}

void ggm_expand_spongent_generic_1t(
    int width_bits, int rounds, int lfsr_bits, uint8_t lfsr_init,
    const uint8_t seed[16], uint32_t depth, uint8_t *out)
{
    int nbytes = width_bits / 8;
    memcpy(out, seed, 16);
    if (depth == 0) return;
    uint64_t total_internal = (1ULL << depth) - 1ULL;
    for (uint64_t i = 0; i < total_internal; i++) {
        const uint8_t *parent = out + i * 16;
        uint8_t in0[SP_MAX_BYTES] = {0};
        uint8_t in1[SP_MAX_BYTES] = {0};
        memcpy(in0, parent, 16);
        memcpy(in1, parent, 16);
        in0[16] = 0x00;
        in1[16] = 0x01;
        uint8_t p0[SP_MAX_BYTES], p1[SP_MAX_BYTES];
        ggm_spongent_block_generic(width_bits, rounds, lfsr_bits, lfsr_init, in0, p0);
        ggm_spongent_block_generic(width_bits, rounds, lfsr_bits, lfsr_init, in1, p1);
        memcpy(out + (2*i + 1) * 16, p0, 16);
        memcpy(out + (2*i + 2) * 16, p1, 16);
    }
}

/* Backward-compatible wrappers for π[176] (existing tests + ctypes). */
void ggm_spongent_pi176_block_ref(const uint8_t in[22], uint8_t out[22]) {
    ggm_spongent_block_generic(176, 80, 7, 0x05, in, out);
}

void ggm_expand_spongent_1t(const uint8_t seed[16], uint32_t depth, uint8_t *out) {
    ggm_expand_spongent_generic_1t(176, 80, 7, 0x05, seed, depth, out);
}
