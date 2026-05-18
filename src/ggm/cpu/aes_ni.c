/* src/ggm/cpu/aes_ni.c — Intel AES-NI variant. Implemented in Phase 7. */
#include "ggmcpu.h"

void ggm_aes128_encrypt_block_ni(const uint8_t key[16],
                                 const uint8_t in[16],
                                 uint8_t out[16]) {
    (void)key; (void)in; (void)out;
    /* Phase 7 implements via _mm_aesenc_si128 intrinsics. */
}

void ggm_expand_aes_ni_1t(const uint8_t seed[16], uint32_t depth, uint8_t *out) {
    (void)seed; (void)depth; (void)out;
}
