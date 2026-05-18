/* src/ggm/cpu/spongent_omp.c — OpenMP-parallel Spongent GGM tree expansion. */
#include "ggmcpu.h"
#include <string.h>
#include <omp.h>

void ggm_expand_spongent_omp(const uint8_t seed[16], uint32_t depth, uint8_t *out, int threads) {
    if (threads > 0) omp_set_num_threads(threads);
    memcpy(out, seed, 16);
    if (depth == 0) return;

    for (uint32_t level = 0; level < depth; level++) {
        uint64_t base_parent = (1ULL << level)       - 1ULL;
        uint64_t base_child  = (1ULL << (level + 1)) - 1ULL;
        uint64_t n           = 1ULL << level;
        #pragma omp parallel for schedule(static)
        for (uint64_t i = 0; i < n; i++) {
            uint8_t in0[22] = {0}, in1[22] = {0};
            memcpy(in0, out + (base_parent + i) * 16, 16);
            memcpy(in1, out + (base_parent + i) * 16, 16);
            in0[16] = 0x00;
            in1[16] = 0x01;
            uint8_t p0[22], p1[22];
            ggm_spongent_pi176_block_ref(in0, p0);
            ggm_spongent_pi176_block_ref(in1, p1);
            memcpy(out + (base_child + 2*i)     * 16, p0, 16);
            memcpy(out + (base_child + 2*i + 1) * 16, p1, 16);
        }
    }
}
