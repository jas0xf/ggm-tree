/* src/ggm/cpu/aes_ni.c — Intel AES-NI variant of the AES-128 PRG. */
#include "ggmcpu.h"
#include <wmmintrin.h>
#include <smmintrin.h>
#include <emmintrin.h>
#include <string.h>

static inline __m128i aes_key_assist(__m128i k, __m128i a) {
    a = _mm_shuffle_epi32(a, 0xff);
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
    return _mm_xor_si128(k, a);
}

static void aes128_key_expand_ni(const uint8_t key[16], __m128i rk[11]) {
    rk[0]  = _mm_loadu_si128((const __m128i *)key);
    rk[1]  = aes_key_assist(rk[0],  _mm_aeskeygenassist_si128(rk[0],  0x01));
    rk[2]  = aes_key_assist(rk[1],  _mm_aeskeygenassist_si128(rk[1],  0x02));
    rk[3]  = aes_key_assist(rk[2],  _mm_aeskeygenassist_si128(rk[2],  0x04));
    rk[4]  = aes_key_assist(rk[3],  _mm_aeskeygenassist_si128(rk[3],  0x08));
    rk[5]  = aes_key_assist(rk[4],  _mm_aeskeygenassist_si128(rk[4],  0x10));
    rk[6]  = aes_key_assist(rk[5],  _mm_aeskeygenassist_si128(rk[5],  0x20));
    rk[7]  = aes_key_assist(rk[6],  _mm_aeskeygenassist_si128(rk[6],  0x40));
    rk[8]  = aes_key_assist(rk[7],  _mm_aeskeygenassist_si128(rk[7],  0x80));
    rk[9]  = aes_key_assist(rk[8],  _mm_aeskeygenassist_si128(rk[8],  0x1b));
    rk[10] = aes_key_assist(rk[9],  _mm_aeskeygenassist_si128(rk[9],  0x36));
}

static inline __m128i aes128_encrypt_ni(__m128i state, const __m128i rk[11]) {
    state = _mm_xor_si128(state, rk[0]);
    state = _mm_aesenc_si128(state, rk[1]);
    state = _mm_aesenc_si128(state, rk[2]);
    state = _mm_aesenc_si128(state, rk[3]);
    state = _mm_aesenc_si128(state, rk[4]);
    state = _mm_aesenc_si128(state, rk[5]);
    state = _mm_aesenc_si128(state, rk[6]);
    state = _mm_aesenc_si128(state, rk[7]);
    state = _mm_aesenc_si128(state, rk[8]);
    state = _mm_aesenc_si128(state, rk[9]);
    return _mm_aesenclast_si128(state, rk[10]);
}

void ggm_aes128_encrypt_block_ni(const uint8_t key[16], const uint8_t in[16], uint8_t out[16]) {
    __m128i rk[11];
    aes128_key_expand_ni(key, rk);
    __m128i ct = aes128_encrypt_ni(_mm_loadu_si128((const __m128i *)in), rk);
    _mm_storeu_si128((__m128i *)out, ct);
}

void ggm_expand_aes_ni_1t(const uint8_t seed[16], uint32_t depth, uint8_t *out) {
    memcpy(out, seed, 16);
    if (depth == 0) return;
    /* Use the same domain-separation inputs as the S-box reference so outputs
     * are bit-exact identical:
     *   ZERO = 0x00 * 16
     *   ONE  = 0x00 * 15 || 0x01      (i.e., byte 15 = 0x01)                  */
    const uint8_t ZERO_b[16] = {0};
    const uint8_t ONE_b[16]  = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
    __m128i ZERO = _mm_loadu_si128((const __m128i *)ZERO_b);
    __m128i ONE  = _mm_loadu_si128((const __m128i *)ONE_b);

    uint64_t total_internal = (1ULL << depth) - 1ULL;
    for (uint64_t i = 0; i < total_internal; i++) {
        __m128i rk[11];
        aes128_key_expand_ni(out + i * 16, rk);
        _mm_storeu_si128((__m128i *)(out + (2*i + 1) * 16), aes128_encrypt_ni(ZERO, rk));
        _mm_storeu_si128((__m128i *)(out + (2*i + 2) * 16), aes128_encrypt_ni(ONE,  rk));
    }
}
