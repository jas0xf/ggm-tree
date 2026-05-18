/* src/ggm/cpu/spongent_ref.c — Spongent-π[176] reference. Implemented in Phase 2. */
#include "ggmcpu.h"
#include <string.h>

void ggm_spongent_pi176_block_ref(const uint8_t in[22], uint8_t out[22]) {
    memcpy(out, in, 22);  /* identity stub; real permutation in Phase 2. */
}

void ggm_expand_spongent_1t(const uint8_t seed[16], uint32_t depth, uint8_t *out) {
    (void)seed; (void)depth; (void)out;
}
