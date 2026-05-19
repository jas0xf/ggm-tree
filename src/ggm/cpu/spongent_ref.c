/* src/ggm/cpu/spongent_ref.c — Spongent-π[176] reference implementation.
 *
 * Mirrors src/ggm/kernels/spongent_kernel.cu exactly: 22-byte state,
 * 80 rounds, PRESENT 4-bit S-box, j → (j * 44) mod 175 bit-permutation
 * (with j=175 fixed), 7-bit LFSR counter (x^7 + x^6 + 1) initialized
 * to 0x05. CPU and GPU produce identical bit-exact output.
 *
 * CHES 2011 spec constants (S-box, pLayer, lCounter polynomial) still
 * need cross-validation against the paper's published π[176](0) vector
 * before declaring the absolute crypto correctness; the CPU/GPU
 * agreement only proves internal consistency.
 */
#include "ggmcpu.h"
#include <string.h>

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

static inline int sp_player(int j) {
    return (j == 175) ? j : (j * 44) % 175;
}

static inline uint8_t sp_lfsr(uint8_t lc) {
    /* 7-bit LFSR with feedback x^7 + x^6 + 1. */
    return (uint8_t)(((lc << 1) | (((lc >> 6) ^ (lc >> 5)) & 1)) & 0x7F);
}

static void sp_round(uint8_t state[22], uint8_t lc) {
    /* AddCounter: lc into low 7 bits, reversed lc shifted left by 1 into bits 169..175. */
    state[0] ^= lc;
    uint8_t rlc = 0;
    for (int k = 0; k < 7; k++) {
        if ((lc >> k) & 1) rlc = (uint8_t)(rlc | (1u << (6 - k)));
    }
    state[21] ^= (uint8_t)(rlc << 1);

    /* S-box layer per nibble. */
    for (int i = 0; i < 22; i++) {
        uint8_t lo = state[i] & 0xF;
        uint8_t hi = (state[i] >> 4) & 0xF;
        state[i] = (uint8_t)((SP4_SBOX[hi] << 4) | SP4_SBOX[lo]);
    }

    /* pLayer via index function. */
    uint8_t buf[22] = {0};
    for (int j = 0; j < 176; j++) {
        sp_setbit(buf, sp_player(j), sp_bit(state, j));
    }
    memcpy(state, buf, 22);
}

void ggm_spongent_pi176_block_ref(const uint8_t in[22], uint8_t out[22]) {
    uint8_t state[22];
    memcpy(state, in, 22);
    uint8_t lc = 0x05;
    for (int r = 0; r < 80; r++) {
        sp_round(state, lc);
        lc = sp_lfsr(lc);
    }
    memcpy(out, state, 22);
}

void ggm_expand_spongent_1t(const uint8_t seed[16], uint32_t depth, uint8_t *out) {
    memcpy(out, seed, 16);
    if (depth == 0) return;
    uint64_t total_internal = (1ULL << depth) - 1ULL;
    for (uint64_t i = 0; i < total_internal; i++) {
        const uint8_t *parent = out + i * 16;
        uint8_t in0[22] = {0}, in1[22] = {0};
        memcpy(in0, parent, 16);
        memcpy(in1, parent, 16);
        in0[16] = 0x00;
        in1[16] = 0x01;
        uint8_t p0[22], p1[22];
        ggm_spongent_pi176_block_ref(in0, p0);
        ggm_spongent_pi176_block_ref(in1, p1);
        memcpy(out + (2*i + 1) * 16, p0, 16);
        memcpy(out + (2*i + 2) * 16, p1, 16);
    }
}
