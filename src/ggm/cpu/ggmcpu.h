/* src/ggm/cpu/ggmcpu.h — C ABI exposed to Python via ctypes. */
#ifndef GGMCPU_H
#define GGMCPU_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* AES-128 single-block primitives. */
void ggm_aes128_encrypt_block_ref(const uint8_t key[16],
                                  const uint8_t in[16],
                                  uint8_t out[16]);
void ggm_aes128_encrypt_block_ni (const uint8_t key[16],
                                  const uint8_t in[16],
                                  uint8_t out[16]);

/* Spongent-π[176] single-block permutation. */
void ggm_spongent_pi176_block_ref(const uint8_t in[22], uint8_t out[22]);

/* GGM tree expansion. `out` must be sized to (2^(depth+1) - 1) * 16 bytes. */
void ggm_expand_aes_sbox_1t  (const uint8_t seed[16], uint32_t depth, uint8_t *out);
void ggm_expand_aes_ni_1t    (const uint8_t seed[16], uint32_t depth, uint8_t *out);
void ggm_expand_aes_sbox_omp (const uint8_t seed[16], uint32_t depth, uint8_t *out, int threads);

void ggm_expand_spongent_1t  (const uint8_t seed[16], uint32_t depth, uint8_t *out);
void ggm_expand_spongent_omp (const uint8_t seed[16], uint32_t depth, uint8_t *out, int threads);

/* Internal helpers exposed so OpenMP wrappers can share the per-block primitives. */
void ggm_aes_key_expansion(const uint8_t key[16], uint8_t rk[176]);
void ggm_aes_encrypt_block(const uint8_t rk[176], const uint8_t in[16], uint8_t out[16]);

#ifdef __cplusplus
}
#endif

#endif /* GGMCPU_H */
