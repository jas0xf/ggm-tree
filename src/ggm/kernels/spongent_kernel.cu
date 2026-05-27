// src/ggm/kernels/spongent_kernel.cu — Generic Spongent GGM expansion.
//
// Parameterized at compile time via PyCUDA defines:
//   SP_WIDTH    — permutation width in bits (136, 176, 240, 272)
//   SP_BYTES    — SP_WIDTH / 8
//   SP_ROUNDS   — number of rounds (70, 80, 100, 140)
//   SP_PMULT    — pLayer multiplier = SP_WIDTH / 4
//   SP_PMOD     — pLayer modulus = SP_WIDTH - 1
//   SP_LFSR_BITS — LFSR width (6, 7, or 8)
//   SP_LFSR_INIT — LFSR initial value
//   SP_HI_BYTE  — byte index for reversed counter = (SP_WIDTH - SP_LFSR_BITS) / 8
//   SP_HI_SHIFT — bit shift for reversed counter = (SP_WIDTH - SP_LFSR_BITS) % 8
//
// Defaults to π[176] if defines are absent (backward compat).

#include <cstdint>

#ifndef SP_WIDTH
#define SP_WIDTH     176
#define SP_BYTES     22
#define SP_ROUNDS    80
#define SP_PMULT     44
#define SP_PMOD      175
#define SP_LFSR_BITS 7
#define SP_LFSR_INIT 0x05
#define SP_HI_BYTE   21
#define SP_HI_SHIFT  1
#endif

extern "C" {

__constant__ uint8_t SP_SBOX[16];
__constant__ uint16_t SP_PLAYER[SP_WIDTH];

}

__device__ __forceinline__ int sp_bit(const uint8_t *s, int i) {
    return (s[i >> 3] >> (i & 7)) & 1;
}

__device__ __forceinline__ void sp_setbit(uint8_t *s, int i, int v) {
    int byte = i >> 3, bit = i & 7;
    s[byte] = (uint8_t)((s[byte] & ~(1u << bit)) | ((v & 1) << bit));
}

__device__ __forceinline__ uint8_t sp_lfsr(uint8_t lc) {
    uint8_t bit = ((lc >> (SP_LFSR_BITS - 1)) ^ (lc >> (SP_LFSR_BITS - 2))) & 1;
    return (uint8_t)(((lc << 1) | bit) & ((1u << SP_LFSR_BITS) - 1));
}

__device__ void sp_round(uint8_t *state, uint8_t lc) {
    // AddCounter
    state[0] ^= lc;
    uint8_t rlc = 0;
    #pragma unroll
    for (int k = 0; k < SP_LFSR_BITS; k++)
        if ((lc >> k) & 1) rlc = (uint8_t)(rlc | (1u << (SP_LFSR_BITS - 1 - k)));
    state[SP_HI_BYTE] ^= (uint8_t)(rlc << SP_HI_SHIFT);

    // S-box layer per nibble
    #pragma unroll
    for (int i = 0; i < SP_BYTES; i++) {
        uint8_t lo = state[i] & 0xF;
        uint8_t hi = (state[i] >> 4) & 0xF;
        state[i] = (uint8_t)((SP_SBOX[hi] << 4) | SP_SBOX[lo]);
    }

    // pLayer via index table
    uint8_t out[SP_BYTES];
    #pragma unroll
    for (int i = 0; i < SP_BYTES; i++) out[i] = 0;
    #pragma unroll 8
    for (int j = 0; j < SP_WIDTH; j++)
        sp_setbit(out, SP_PLAYER[j], sp_bit(state, j));
    #pragma unroll
    for (int i = 0; i < SP_BYTES; i++) state[i] = out[i];
}

__device__ void sp_permute(const uint8_t *in, uint8_t *out) {
    uint8_t s[SP_BYTES];
    #pragma unroll
    for (int i = 0; i < SP_BYTES; i++) s[i] = in[i];
    uint8_t lc = SP_LFSR_INIT;
    #pragma unroll 8
    for (int r = 0; r < SP_ROUNDS; r++) {
        sp_round(s, lc);
        lc = sp_lfsr(lc);
    }
    #pragma unroll
    for (int i = 0; i < SP_BYTES; i++) out[i] = s[i];
}

extern "C" __global__
void ggm_spongent_expand_level(uint8_t *tree, uint32_t level) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t level_size = 1ULL << level;
    if (i >= level_size) return;

    const uint8_t *p_src = tree + ((level_size - 1ULL) + i) * 16;
    uint8_t in0[SP_BYTES];
    uint8_t in1[SP_BYTES];
    #pragma unroll
    for (int k = 0; k < SP_BYTES; k++) { in0[k] = 0; in1[k] = 0; }
    #pragma unroll
    for (int k = 0; k < 16; k++) { in0[k] = p_src[k]; in1[k] = p_src[k]; }
    in0[16] = 0x00;
    in1[16] = 0x01;

    uint8_t p0[SP_BYTES], p1[SP_BYTES];
    sp_permute(in0, p0);
    sp_permute(in1, p1);

    uint64_t base = ((1ULL << (level + 1)) - 1ULL) + 2ULL * i;
    uint8_t *dst0 = tree + base * 16;
    uint8_t *dst1 = tree + (base + 1) * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) dst0[k] = p0[k];
    #pragma unroll
    for (int k = 0; k < 16; k++) dst1[k] = p1[k];
}
