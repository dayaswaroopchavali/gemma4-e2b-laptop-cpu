// W4A8 AVX2 kernel for prefill (BOUNDED: activations rounded to int8 per 32-group).
// Weights are repacked at load into 8-row tiles: for tile t, k-chunk c (8 k's): 32 bytes where bytes [4i..4i+3]
// hold row i; low nibbles = k 8c..8c+3, high nibbles = k 8c+4..8c+7.  One vpmaddubsw+vpmaddwd then sums 4 k per
// output-row lane, so a 32-byte weight load feeds 8 rows x 8 k for every activation row (M rows share it).
//   y[m][n] = sum_g sw[n,g] * sx[m,g] * ( sum_{i in g} q[n,i]*xq[m,i]  -  8 * sumxq[m,g] ),   q in 0..15 (=signed+8)
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace w4a8 {
inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

// original layout: W[n][k/2] bytes (low nibble = even k), scales S[n][G] bf16  ->  tiles
struct Packed { int N, K; std::vector<uint8_t> w; std::vector<uint16_t> s; };   // s layout: [tile][g][8 rows]
inline void repack_alloc(int N, int K, Packed& P) { P.N = N; P.K = K; P.w.assign((size_t)(N / 8) * (K / 8) * 32, 0); P.s.assign((size_t)(N / 8) * (K / 32) * 8, 0); }
inline void repack_tiles(const uint8_t* W, const uint16_t* S, int N, int K, Packed& P, int t0, int t1) {
  const int G = K / 32, KC = K / 8;
  for (int t = t0; t < t1; ++t) for (int c = 0; c < KC; ++c) for (int i = 0; i < 8; ++i) for (int j = 0; j < 4; ++j) {
      int n = t * 8 + i; int ke = 8 * c + j, ko = 8 * c + 4 + j;                      // elements for low / high nibble
      auto nib = [&](int k) { uint8_t b = W[(size_t)n * (K / 2) + k / 2]; return (k & 1) ? (b >> 4) : (b & 15); };
      P.w[((size_t)t * KC + c) * 32 + i * 4 + j] = (uint8_t)(nib(ke) | (nib(ko) << 4)); }
  for (int t = t0; t < t1; ++t) for (int g = 0; g < G; ++g) for (int i = 0; i < 8; ++i) P.s[((size_t)t * G + g) * 8 + i] = S[(size_t)(t * 8 + i) * G + g];
}
// quantise M activation rows (fp32, K each) to int8 per 32-group (AVX2; round-to-nearest-even)
struct XQ { std::vector<int8_t> q; std::vector<float> sx; std::vector<int32_t> sum; int M = 0, K = 0; };
inline void quant(const float* x, int M, int K, XQ& X) {
  X.M = M; X.K = K; const int G = K / 32; if (X.q.size() < (size_t)M * K) X.q.resize((size_t)M * K); if (X.sx.size() < (size_t)M * G) { X.sx.resize((size_t)M * G); X.sum.resize((size_t)M * G); }
  const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  for (int m = 0; m < M; ++m) for (int g = 0; g < G; ++g) { const float* p = x + (size_t)m * K + g * 32;
    __m256 a0 = _mm256_loadu_ps(p), a1 = _mm256_loadu_ps(p + 8), a2 = _mm256_loadu_ps(p + 16), a3 = _mm256_loadu_ps(p + 24);
    __m256 mx = _mm256_max_ps(_mm256_max_ps(_mm256_and_ps(a0, absmask), _mm256_and_ps(a1, absmask)), _mm256_max_ps(_mm256_and_ps(a2, absmask), _mm256_and_ps(a3, absmask)));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1)); m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4)); m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    float amax = _mm_cvtss_f32(m4), sx = amax / 127.0f, inv = sx > 0 ? 1.0f / sx : 0.0f; __m256 vi = _mm256_set1_ps(inv);
    __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(a0, vi)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(a1, vi)), i2 = _mm256_cvtps_epi32(_mm256_mul_ps(a2, vi)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(a3, vi));
    __m256i s = _mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)); __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1)); s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E)); s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
    __m256i p01 = _mm256_packs_epi32(i0, i1), p23 = _mm256_packs_epi32(i2, i3); __m256i pb = _mm256_packs_epi16(p01, p23); pb = _mm256_permutevar8x32_epi32(pb, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256((__m256i*)(X.q.data() + (size_t)m * K + g * 32), pb); X.sx[(size_t)m * G + g] = sx; X.sum[(size_t)m * G + g] = _mm_cvtsi128_si32(s4); }
}
static inline __m256 widen_bf(const uint16_t* p) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)), 16)); }
// one 8-row tile, MT activation rows starting at row m0: writes Y[m][n0..n0+7] (ldy = N)
template <int MT>
inline void tile(const Packed& P, int t, const XQ& X, int m0, float* Y) {
  const int G = P.K / 32, KC = P.K / 8; const uint8_t* wt = P.w.data() + (size_t)t * KC * 32; const uint16_t* st = P.s.data() + (size_t)t * G * 8;
  const __m256i ones = _mm256_set1_epi16(1), m15 = _mm256_set1_epi8(0x0F); __m256 row[MT];
  for (int m = 0; m < MT; ++m) row[m] = _mm256_setzero_ps();
  for (int g = 0; g < G; ++g) {
    __m256i acc[MT]; for (int m = 0; m < MT; ++m) acc[m] = _mm256_setzero_si256();
    for (int cc = 0; cc < 4; ++cc) { __m256i w = _mm256_loadu_si256((const __m256i*)(wt + ((size_t)g * 4 + cc) * 32));
      __m256i lo = _mm256_and_si256(w, m15), hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), m15);
      for (int m = 0; m < MT; ++m) { const int8_t* xq = X.q.data() + (size_t)(m0 + m) * P.K + g * 32 + cc * 8; int32_t a, b; std::memcpy(&a, xq, 4); std::memcpy(&b, xq + 4, 4);
        __m256i p = _mm256_add_epi16(_mm256_maddubs_epi16(lo, _mm256_set1_epi32(a)), _mm256_maddubs_epi16(hi, _mm256_set1_epi32(b))); acc[m] = _mm256_add_epi16(acc[m], p); } }
    __m256 sw = widen_bf(st + (size_t)g * 8);
    for (int m = 0; m < MT; ++m) { __m256i a32 = _mm256_madd_epi16(acc[m], ones); a32 = _mm256_sub_epi32(a32, _mm256_set1_epi32(8 * X.sum[(size_t)(m0 + m) * G + g]));
      __m256 sc = _mm256_mul_ps(sw, _mm256_set1_ps(X.sx[(size_t)(m0 + m) * G + g])); row[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a32), sc, row[m]); }
  }
  for (int m = 0; m < MT; ++m) _mm256_storeu_ps(Y + (size_t)(m0 + m) * P.N + t * 8, row[m]);
}
}  // namespace w4a8
