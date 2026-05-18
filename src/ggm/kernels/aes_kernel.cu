// src/ggm/kernels/aes_kernel.cu — variable-key AES-128 GGM expansion kernels.
//
// Two entry points compiled into one PyCUDA module:
//   ggm_aes_sbox_expand_level   — S-box-only AES, one parent per thread
//   ggm_aes_ttable_expand_level — T-tables in shared memory, one parent per thread
//
// Both implement G(s) = AES_s(0) || AES_s(1), where each thread reads
// tree[(1<<level)-1 + i] (parent) and writes the two children at
// tree[(1<<(level+1))-1 + 2*i + {0,1}].
//
// Constants SBOX_C, RCON_C, T0_C..T3_C are uploaded by host.py before launch.

#include <cstdint>

extern "C" {

__constant__ uint8_t  SBOX_C[256];
__constant__ uint8_t  RCON_C[11];
__constant__ uint32_t T0_C[256];
__constant__ uint32_t T1_C[256];
__constant__ uint32_t T2_C[256];
__constant__ uint32_t T3_C[256];

}

__device__ __forceinline__ uint8_t xtime_d(uint8_t a) {
    return (uint8_t)((a << 1) ^ (((a >> 7) & 1) * 0x1b));
}

__device__ void key_expansion_d(const uint8_t key[16], uint8_t rk[176]) {
    #pragma unroll
    for (int i = 0; i < 16; i++) rk[i] = key[i];
    for (int i = 16; i < 176; i += 4) {
        uint8_t t0 = rk[i - 4], t1 = rk[i - 3], t2 = rk[i - 2], t3 = rk[i - 1];
        if ((i & 0xF) == 0) {
            uint8_t r = t0;
            t0 = SBOX_C[t1] ^ RCON_C[i >> 4];
            t1 = SBOX_C[t2];
            t2 = SBOX_C[t3];
            t3 = SBOX_C[r];
        }
        rk[i]     = rk[i - 16]     ^ t0;
        rk[i + 1] = rk[i - 16 + 1] ^ t1;
        rk[i + 2] = rk[i - 16 + 2] ^ t2;
        rk[i + 3] = rk[i - 16 + 3] ^ t3;
    }
}

__device__ void aes_block_sbox_d(const uint8_t rk[176],
                                 const uint8_t in[16],
                                 uint8_t out[16]) {
    uint8_t s[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) s[i] = in[i] ^ rk[i];
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
            s[4*c]     = (uint8_t)(b0 ^ a3 ^ a2 ^ b1 ^ a1) ^ rk[16*r + 4*c];
            s[4*c + 1] = (uint8_t)(b1 ^ a0 ^ a3 ^ b2 ^ a2) ^ rk[16*r + 4*c + 1];
            s[4*c + 2] = (uint8_t)(b2 ^ a1 ^ a0 ^ b3 ^ a3) ^ rk[16*r + 4*c + 2];
            s[4*c + 3] = (uint8_t)(b3 ^ a2 ^ a1 ^ b0 ^ a0) ^ rk[16*r + 4*c + 3];
        }
    }
    uint8_t t[16];
    t[0]  = SBOX_C[s[0]];  t[1]  = SBOX_C[s[5]];  t[2]  = SBOX_C[s[10]]; t[3]  = SBOX_C[s[15]];
    t[4]  = SBOX_C[s[4]];  t[5]  = SBOX_C[s[9]];  t[6]  = SBOX_C[s[14]]; t[7]  = SBOX_C[s[3]];
    t[8]  = SBOX_C[s[8]];  t[9]  = SBOX_C[s[13]]; t[10] = SBOX_C[s[2]];  t[11] = SBOX_C[s[7]];
    t[12] = SBOX_C[s[12]]; t[13] = SBOX_C[s[1]];  t[14] = SBOX_C[s[6]];  t[15] = SBOX_C[s[11]];
    #pragma unroll
    for (int i = 0; i < 16; i++) out[i] = t[i] ^ rk[160 + i];
}

extern "C" __global__
void ggm_aes_sbox_expand_level(uint8_t *tree, uint32_t level) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t level_size = 1ULL << level;
    if (i >= level_size) return;

    uint8_t parent[16];
    const uint8_t *p_src = tree + ((level_size - 1ULL) + i) * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) parent[k] = p_src[k];

    uint8_t rk[176];
    key_expansion_d(parent, rk);

    uint8_t zero[16] = {0};
    uint8_t one[16]  = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1};

    uint64_t base = ((1ULL << (level + 1)) - 1ULL) + 2ULL * i;
    aes_block_sbox_d(rk, zero, tree + base * 16);
    aes_block_sbox_d(rk, one,  tree + (base + 1) * 16);
}

// ---------------------------- T-tables variant -----------------------------
//
// T-tables encode SubBytes + ShiftRows + MixColumns. Each table maps
// a byte to a 32-bit column contribution. We load 4 KB of tables into
// shared memory once per block at kernel entry.

__device__ __forceinline__ uint32_t pack_le32(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
    return ((uint32_t)b0)
         | ((uint32_t)b1 << 8)
         | ((uint32_t)b2 << 16)
         | ((uint32_t)b3 << 24);
}

__device__ void aes_block_tt_d(const uint8_t  rk[176],
                               const uint32_t T0[256],
                               const uint32_t T1[256],
                               const uint32_t T2[256],
                               const uint32_t T3[256],
                               const uint8_t  in[16],
                               uint8_t        out[16]) {
    uint32_t s0 = pack_le32(in[0], in[1], in[2], in[3])
                ^ pack_le32(rk[0], rk[1], rk[2], rk[3]);
    uint32_t s1 = pack_le32(in[4], in[5], in[6], in[7])
                ^ pack_le32(rk[4], rk[5], rk[6], rk[7]);
    uint32_t s2 = pack_le32(in[8], in[9], in[10], in[11])
                ^ pack_le32(rk[8], rk[9], rk[10], rk[11]);
    uint32_t s3 = pack_le32(in[12], in[13], in[14], in[15])
                ^ pack_le32(rk[12], rk[13], rk[14], rk[15]);

    #pragma unroll
    for (int r = 1; r <= 9; r++) {
        uint32_t t0 = T0[ s0        & 0xff]
                    ^ T1[(s1 >>  8) & 0xff]
                    ^ T2[(s2 >> 16) & 0xff]
                    ^ T3[(s3 >> 24) & 0xff]
                    ^ pack_le32(rk[16*r + 0], rk[16*r + 1], rk[16*r + 2], rk[16*r + 3]);
        uint32_t t1 = T0[ s1        & 0xff]
                    ^ T1[(s2 >>  8) & 0xff]
                    ^ T2[(s3 >> 16) & 0xff]
                    ^ T3[(s0 >> 24) & 0xff]
                    ^ pack_le32(rk[16*r + 4], rk[16*r + 5], rk[16*r + 6], rk[16*r + 7]);
        uint32_t t2 = T0[ s2        & 0xff]
                    ^ T1[(s3 >>  8) & 0xff]
                    ^ T2[(s0 >> 16) & 0xff]
                    ^ T3[(s1 >> 24) & 0xff]
                    ^ pack_le32(rk[16*r + 8], rk[16*r + 9], rk[16*r + 10], rk[16*r + 11]);
        uint32_t t3 = T0[ s3        & 0xff]
                    ^ T1[(s0 >>  8) & 0xff]
                    ^ T2[(s1 >> 16) & 0xff]
                    ^ T3[(s2 >> 24) & 0xff]
                    ^ pack_le32(rk[16*r + 12], rk[16*r + 13], rk[16*r + 14], rk[16*r + 15]);
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    // Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns).
    uint8_t b[16];
    b[0]  = SBOX_C[ s0        & 0xff]; b[1]  = SBOX_C[(s1 >>  8) & 0xff];
    b[2]  = SBOX_C[(s2 >> 16) & 0xff]; b[3]  = SBOX_C[(s3 >> 24) & 0xff];
    b[4]  = SBOX_C[ s1        & 0xff]; b[5]  = SBOX_C[(s2 >>  8) & 0xff];
    b[6]  = SBOX_C[(s3 >> 16) & 0xff]; b[7]  = SBOX_C[(s0 >> 24) & 0xff];
    b[8]  = SBOX_C[ s2        & 0xff]; b[9]  = SBOX_C[(s3 >>  8) & 0xff];
    b[10] = SBOX_C[(s0 >> 16) & 0xff]; b[11] = SBOX_C[(s1 >> 24) & 0xff];
    b[12] = SBOX_C[ s3        & 0xff]; b[13] = SBOX_C[(s0 >>  8) & 0xff];
    b[14] = SBOX_C[(s1 >> 16) & 0xff]; b[15] = SBOX_C[(s2 >> 24) & 0xff];
    #pragma unroll
    for (int k = 0; k < 16; k++) out[k] = b[k] ^ rk[160 + k];
}

extern "C" __global__
void ggm_aes_ttable_expand_level(uint8_t *tree, uint32_t level) {
    __shared__ uint32_t T0[256];
    __shared__ uint32_t T1[256];
    __shared__ uint32_t T2[256];
    __shared__ uint32_t T3[256];
    int tid = threadIdx.x;
    for (int k = tid; k < 256; k += blockDim.x) {
        T0[k] = T0_C[k]; T1[k] = T1_C[k]; T2[k] = T2_C[k]; T3[k] = T3_C[k];
    }
    __syncthreads();

    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t level_size = 1ULL << level;
    if (i >= level_size) return;

    uint8_t parent[16];
    const uint8_t *p_src = tree + ((level_size - 1ULL) + i) * 16;
    #pragma unroll
    for (int k = 0; k < 16; k++) parent[k] = p_src[k];

    uint8_t rk[176];
    key_expansion_d(parent, rk);

    uint8_t zero[16] = {0};
    uint8_t one[16]  = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1};

    uint64_t base = ((1ULL << (level + 1)) - 1ULL) + 2ULL * i;
    aes_block_tt_d(rk, T0, T1, T2, T3, zero, tree + base * 16);
    aes_block_tt_d(rk, T0, T1, T2, T3, one,  tree + (base + 1) * 16);
}

// ---------------------- Single-leaf path evaluation ------------------------

extern "C" __global__
void ggm_aes_sbox_path_eval(const uint8_t *seed_in,
                            uint64_t       path_idx,
                            uint32_t       depth,
                            uint8_t       *out_leaf) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint8_t cur[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) cur[k] = seed_in[k];

    for (uint32_t lvl = 0; lvl < depth; lvl++) {
        uint8_t rk[176];
        key_expansion_d(cur, rk);
        uint64_t bit = (path_idx >> (depth - 1 - lvl)) & 1ULL;
        uint8_t in[16] = {0};
        if (bit) in[15] = 0x01;
        uint8_t nxt[16];
        aes_block_sbox_d(rk, in, nxt);
        #pragma unroll
        for (int k = 0; k < 16; k++) cur[k] = nxt[k];
    }
    #pragma unroll
    for (int k = 0; k < 16; k++) out_leaf[k] = cur[k];
}
