/* src/ggm/cpu/aes_omp.c — OpenMP-parallel AES tree expansion. Implemented in Phase 7. */
#include "ggmcpu.h"

void ggm_expand_aes_sbox_omp(const uint8_t seed[16], uint32_t depth, uint8_t *out, int threads) {
    (void)seed; (void)depth; (void)out; (void)threads;
}
