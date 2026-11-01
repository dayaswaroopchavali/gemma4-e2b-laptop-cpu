// AVX2 W4A32 GEMV for symmetric int4, group size 32, on Zen3 (no AVX-512).
// Weight layout = compressed-tensors/vLLM packing: row n holds K/8 int32; element j of a
// group of 8 sits in bits [4j, 4j+4) as (q+8). scales: [N, K/32] bf16.
//   y[n] = sum_g s[n,g] * ( sum_{i in g} q[n,i]*x[i]  -  8 * sum_{i in g} x[i] )
// Low nibbles are even elements, high nibbles odd, so x is split once into even/odd lanes
// and every row reuses that split and the per-group sums of x (identity proved in Lean).
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace w4 {

inline float bf16_to_f32(uint16_t b) {
  uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f;
}

// xe/xo: [K/2] even/odd split of x; gsum: [K/32] sums of x per group.
inline void prep_x(const float* x, int K, float* xe, float* xo, float* gsum) {
  for (int i = 0; i < K / 2; ++i) { xe[i] = x[2 * i]; xo[i] = x[2 * i + 1]; }
  for (int g = 0; g < K / 32; ++g) { float s = 0; for (int i = 0; i < 32; ++i) s += x[g * 32 + i]; gsum[g] = s; }
}

static inline __m256 dot_group(const uint8_t* p, const float* xe, const float* xo, __m256 acc) {
  __m128i v = _mm_loadu_si128((const __m128i*)p);                 // 16 bytes = 32 nibbles
  __m128i m = _mm_set1_epi8(0x0F);
  __m128i lo = _mm_and_si128(v, m);                                // even elements
  __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), m);             // odd elements
  __m256 l0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
  __m256 l1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
  __m256 h0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
  __m256 h1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));
  acc = _mm256_fmadd_ps(l0, _mm256_loadu_ps(xe), acc);
  acc = _mm256_fmadd_ps(l1, _mm256_loadu_ps(xe + 8), acc);
  acc = _mm256_fmadd_ps(h0, _mm256_loadu_ps(xo), acc);
  acc = _mm256_fmadd_ps(h1, _mm256_loadu_ps(xo + 8), acc);
  return acc;
}

static inline float hsum(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_hadd_ps(a, a); a = _mm_hadd_ps(a, a); return _mm_cvtss_f32(a);
}

// y[n] for n in [n0, n1)
inline void gemv_rows(const uint8_t* W, const uint16_t* S, const float* xe, const float* xo,
                      const float* gsum, int K, int n0, int n1, float* y) {
  const int G = K / 32; const size_t rb = (size_t)K / 2;
  for (int n = n0; n < n1; ++n) {
    const uint8_t* w = W + (size_t)n * rb; const uint16_t* s = S + (size_t)n * G;
    __m256 row = _mm256_setzero_ps(); float off = 0.f;
    for (int g = 0; g < G; ++g) {
      if (g + 8 < G) _mm_prefetch((const char*)(w + (size_t)(g + 8) * 16), _MM_HINT_T0);
      __m256 a = dot_group(w + (size_t)g * 16, xe + g * 16, xo + g * 16, _mm256_setzero_ps());
      float sc = bf16_to_f32(s[g]);
      row = _mm256_fmadd_ps(a, _mm256_set1_ps(sc), row);
      off += sc * gsum[g];
    }
    y[n] = hsum(row) - 8.f * off;
  }
}

// MT activations at once for rows [n0,n1): Y[m*ldy + n]
template <int MT>
inline void gemm_rows_tile(const uint8_t* W, const uint16_t* S, const float* xe, const float* xo,
                           const float* gsum, int K, int n0, int n1, float* Y, int ldy, int ldx_h,
                           int ldg) {
  const int G = K / 32; const size_t rb = (size_t)K / 2;
  for (int n = n0; n < n1; ++n) {
    const uint8_t* w = W + (size_t)n * rb; const uint16_t* s = S + (size_t)n * G;
    __m256 row[MT]; float off[MT];
    for (int m = 0; m < MT; ++m) { row[m] = _mm256_setzero_ps(); off[m] = 0.f; }
    for (int g = 0; g < G; ++g) {
      if (g + 16 < G) _mm_prefetch((const char*)(w + (size_t)(g + 16) * 16), _MM_HINT_NTA);   // weights are streamed once per forward: keep them out of L2/L3 so the activations stay resident
      __m128i v = _mm_loadu_si128((const __m128i*)(w + (size_t)g * 16));
      __m128i mk = _mm_set1_epi8(0x0F);
      __m128i lo = _mm_and_si128(v, mk), hi = _mm_and_si128(_mm_srli_epi16(v, 4), mk);
      __m256 l0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
      __m256 l1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
      __m256 h0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
      __m256 h1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));
      float sc = bf16_to_f32(s[g]); __m256 scv = _mm256_set1_ps(sc);
      for (int m = 0; m < MT; ++m) {
        const float* pe = xe + (size_t)m * ldx_h + g * 16; const float* po = xo + (size_t)m * ldx_h + g * 16;
        __m256 a = _mm256_mul_ps(l0, _mm256_loadu_ps(pe));
        a = _mm256_fmadd_ps(l1, _mm256_loadu_ps(pe + 8), a);
        a = _mm256_fmadd_ps(h0, _mm256_loadu_ps(po), a);
        a = _mm256_fmadd_ps(h1, _mm256_loadu_ps(po + 8), a);
        row[m] = _mm256_fmadd_ps(a, scv, row[m]);
        off[m] += sc * gsum[(size_t)m * ldg + g];
      }
    }
    for (int m = 0; m < MT; ++m) Y[(size_t)m * ldy + n] = hsum(row[m]) - 8.f * off[m];
  }
}

// bf16 activations straight to the even/odd split + group sums (same arithmetic as prep_x on the
// bf16->fp32 widened values, so results are bit-identical to the widen-then-prep path).
inline void prep_x_bf16(const uint16_t* x, int K, float* xe, float* xo, float* gsum) {
  for (int i = 0; i < K / 2; ++i) { xe[i] = bf16_to_f32(x[2 * i]); xo[i] = bf16_to_f32(x[2 * i + 1]); }
  for (int g = 0; g < K / 32; ++g) { float s = 0; for (int i = 0; i < 32; ++i) s += bf16_to_f32(x[g * 32 + i]); gsum[g] = s; }
}

inline void gemv(const uint8_t* W, const uint16_t* S, const float* x, int N, int K, float* y) {
  std::vector<float> xe(K / 2), xo(K / 2), gs(K / 32);
  prep_x(x, K, xe.data(), xo.data(), gs.data());
  #pragma omp parallel
  {
    int t = omp_get_thread_num(), T = omp_get_num_threads();
    int per = (N + T - 1) / T; int a = t * per, b = a + per < N ? a + per : N;
    if (a < b) gemv_rows(W, S, xe.data(), xo.data(), gs.data(), K, a, b, y);
  }
}
}  // namespace w4
