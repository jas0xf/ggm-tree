// src/ggm/kernels/spongent_kernel.cu — Spongent-π[176] GGM expansion.
//
// G(s) = π[176](s || 0x00)[0..15] || π[176](s || 0x01)[0..15]
//
// Per node, two π calls with byte-16 domain separation, then truncate the
// 22-byte permutation output to 16 bytes for the child.

#include <cstdint>

extern "C" {

__constant__ uint8_t SP_SBOX[16];      // PRESENT 4-bit S-box
__constant__ uint8_t SP_PLAYER[176];   // destination bit index per source bit

}

__device__ __forceinline__ int sp_bit(const uint8_t *s, int i) {
    return (s[i >> 3] >> (i & 7)) & 1;
}

__device__ __forceinline__ void sp_setbit(uint8_t *s, int i, int v) {
    int byte = i >> 3, bit = i & 7;
    s[byte] = (uint8_t)((s[byte] & ~(1u << bit)) | ((v & 1) << bit));
}

__device__ __forceinline__ uint8_t sp_lfsr(uint8_t lc) {
    // 7-bit LFSR with feedback x^7 + x^6 + 1
    return (uint8_t)(((lc << 1) | (((lc >> 6) ^ (lc >> 5)) & 1)) & 0x7F);
}

__device__ void sp_round(uint8_t *state, uint8_t lc) {
    // AddCounter — XOR lCounter into low 7 bits, reversed lCounter into bits 169..175
    state[0] ^= lc;
    uint8_t rlc = 0;
    #pragma unroll
    for (int k = 0; k < 7; k++) if ((lc >> k) & 1) rlc = (uint8_t)(rlc | (1u << (6 - k)));
    state[21] ^= (uint8_t)(rlc << 1);
    // S-box layer per nibble
    #pragma unroll
    for (int i = 0; i < 22; i++) {
        uint8_t lo = state[i] & 0xF;
        uint8_t hi = (state[i] >> 4) & 0xF;
        state[i] = (uint8_t)((SP_SBOX[hi] << 4) | SP_SBOX[lo]);
    }
    // pLayer via index table
    uint8_t out[22] = {0};
    #pragma unroll
    for (int j = 0; j < 176; j++) sp_setbit(out, SP_PLAYER[j], sp_bit(state, j));
    #pragma unroll
    for (int i = 0; i < 22; i++) state[i] = out[i];
}

__device__ void sp_pi176(const uint8_t in[22], uint8_t out[22]) {
    uint8_t s[22];
    #pragma unroll
    for (int i = 0; i < 22; i++) s[i] = in[i];
    uint8_t lc = 0x05;
    #pragma unroll 8
    for (int r = 0; r < 80; r++) {
        sp_round(s, lc);
        lc = sp_lfsr(lc);
    }
    #pragma unroll
    for (int i = 0; i < 22; i++) out[i] = s[i];
}

extern "C" __global__
void ggm_spongent_expand_level(uint8_t *tree, uint32_t level) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t level_size = 1ULL << level;
    if (i >= level_size) return;

    const uint8_t *p_src = tree + ((level_size - 1ULL) + i) * 16;
    uint8_t in0[22] = {0}, in1[22] = {0};
    #pragma unroll
    for (int k = 0; k < 16; k++) { in0[k] = p_src[k]; in1[k] = p_src[k]; }
    in0[16] = 0x00;
    in1[16] = 0x01;

    uint8_t p0[22], p1[22];
    sp_pi176(in0, p0);
    sp_pi176(in1, p1);

    uint64_t base = ((1ULL << (level + 1)) - 1ULL) + 2ULL * i;
    uint8_t *dst0 = tree + base * 16;
    uint8_t *dst1 = tree + (base + 1) * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) dst0[k] = p0[k];
    #pragma unroll
    for (int k = 0; k < 16; k++) dst1[k] = p1[k];
}
