// src/ggm/kernels/aes_kernel_fixedkey.cu — fixed-key AES-128 GGM expansion.
//
// G(s) = AES_K(s)        || AES_K(s ⊕ 0x01)
// where the public 128-bit K has its expanded round-key schedule precomputed
// once on the host and uploaded into constant memory as FIXED_RK[176].
// Domain separation between the two children: bit-flip on byte 0.

#include <cstdint>

extern "C" {

__constant__ uint8_t SBOX_C[256];     // shared with variable-key module via separate upload
__constant__ uint8_t FIXED_RK[176];   // expanded round keys for the public K

}

__device__ __forceinline__ uint8_t xtime_d(uint8_t a) {
    return (uint8_t)((a << 1) ^ (((a >> 7) & 1) * 0x1b));
}

__device__ void aes_block_fk_d(const uint8_t in[16], uint8_t out[16]) {
    uint8_t s[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) s[i] = in[i] ^ FIXED_RK[i];
    #pragma unroll
    for (int r = 1; r <= 9; r++) {
        uint8_t t[16];
        t[0]  = SBOX_C[s[0]];  t[1]  = SBOX_C[s[5]];  t[2]  = SBOX_C[s[10]]; t[3]  = SBOX_C[s[15]];
        t[4]  = SBOX_C[s[4]];  t[5]  = SBOX_C[s[9]];  t[6]  = SBOX_C[s[14]]; t[7]  = SBOX_C[s[3]];
        t[8]  = SBOX_C[s[8]];  t[9]  = SBOX_C[s[13]]; t[10] = SBOX_C[s[2]];  t[11] = SBOX_C[s[7]];
        t[12] = SBOX_C[s[12]]; t[13] = SBOX_C[s[1]];  t[14] = SBOX_C[s[6]];  t[15] = SBOX_C[s[11]];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            uint8_t a0 = t[4*c], a1 = t[4*c+1], a2 = t[4*c+2], a3 = t[4*c+3];
            uint8_t b0 = xtime_d(a0), b1 = xtime_d(a1), b2 = xtime_d(a2), b3 = xtime_d(a3);
            s[4*c]     = (uint8_t)(b0 ^ a3 ^ a2 ^ b1 ^ a1) ^ FIXED_RK[16*r + 4*c];
            s[4*c + 1] = (uint8_t)(b1 ^ a0 ^ a3 ^ b2 ^ a2) ^ FIXED_RK[16*r + 4*c + 1];
            s[4*c + 2] = (uint8_t)(b2 ^ a1 ^ a0 ^ b3 ^ a3) ^ FIXED_RK[16*r + 4*c + 2];
            s[4*c + 3] = (uint8_t)(b3 ^ a2 ^ a1 ^ b0 ^ a0) ^ FIXED_RK[16*r + 4*c + 3];
        }
    }
    uint8_t t[16];
    t[0]  = SBOX_C[s[0]];  t[1]  = SBOX_C[s[5]];  t[2]  = SBOX_C[s[10]]; t[3]  = SBOX_C[s[15]];
    t[4]  = SBOX_C[s[4]];  t[5]  = SBOX_C[s[9]];  t[6]  = SBOX_C[s[14]]; t[7]  = SBOX_C[s[3]];
    t[8]  = SBOX_C[s[8]];  t[9]  = SBOX_C[s[13]]; t[10] = SBOX_C[s[2]];  t[11] = SBOX_C[s[7]];
    t[12] = SBOX_C[s[12]]; t[13] = SBOX_C[s[1]];  t[14] = SBOX_C[s[6]];  t[15] = SBOX_C[s[11]];
    #pragma unroll
    for (int i = 0; i < 16; i++) out[i] = t[i] ^ FIXED_RK[160 + i];
}

extern "C" __global__
void ggm_aes_sbox_fixedkey_expand_level(uint8_t *tree, uint32_t level) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t level_size = 1ULL << level;
    if (i >= level_size) return;

    uint8_t parent[16];
    const uint8_t *p_src = tree + ((level_size - 1ULL) + i) * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) parent[k] = p_src[k];

    uint8_t in0[16], in1[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) { in0[k] = parent[k]; in1[k] = parent[k]; }
    in1[0] ^= 0x01;  // domain separation

    uint64_t base = ((1ULL << (level + 1)) - 1ULL) + 2ULL * i;
    aes_block_fk_d(in0, tree + base * 16);
    aes_block_fk_d(in1, tree + (base + 1) * 16);
}
