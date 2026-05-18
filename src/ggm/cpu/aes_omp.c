/* src/ggm/cpu/aes_omp.c — OpenMP-parallel S-box AES-128 GGM tree expansion. */
#include "ggmcpu.h"
#include <string.h>
#include <omp.h>

void ggm_expand_aes_sbox_omp(const uint8_t seed[16], uint32_t depth, uint8_t *out, int threads) {
    if (threads > 0) omp_set_num_threads(threads);
    memcpy(out, seed, 16);
    if (depth == 0) return;

    static const uint8_t ZERO[16] = {0};
    static const uint8_t ONE[16]  = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};

    for (uint32_t level = 0; level < depth; level++) {
        uint64_t base_parent = (1ULL << level)       - 1ULL;
        uint64_t base_child  = (1ULL << (level + 1)) - 1ULL;
        uint64_t n           = 1ULL << level;
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n; i++) {
            uint8_t rk[176];
            ggm_aes_key_expansion(out + (base_parent + i) * 16, rk);
            ggm_aes_encrypt_block(rk, ZERO, out + (base_child + 2*i)     * 16);
            ggm_aes_encrypt_block(rk, ONE,  out + (base_child + 2*i + 1) * 16);
        }
    }
}
