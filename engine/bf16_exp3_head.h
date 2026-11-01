// Lossless exponent-packed BF16 matrix (K multiple of 32), 11 bits/weight + exact correction list.
// Per group of 32 weights: 32 B (sign<<7 | mant7), then three 4-B planes holding bit k of the 3-bit
// exponent index of weight j at bit j.  Exceptions (exponent not in the 8-entry codebook) are
// stored with the nearest codebook index and corrected by  sum (w - decoded) * x  (patch_exact).
#pragma once
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace bf16p {

struct Packed {
  int N = 0, K = 0;
  uint8_t cb[16] = {0};                 // codebook: index -> exponent
  std::vector<uint8_t> data;            // N * (K/32) * 44 bytes
  std::vector<int64_t> exc_ptr;         // CSR over rows
  std::vector<uint16_t> exc_col;
  std::vector<float> exc_delta;         // (true - decoded) as exact fp32
};

inline float bf16f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

inline void pack(const uint16_t* W, int N, int K, const uint8_t cb8[8], Packed& P) {
  P.N = N; P.K = K; std::memcpy(P.cb, cb8, 8);
  const int G = K / 32; P.data.assign((size_t)N * G * 44, 0); P.exc_ptr.assign(N + 1, 0);
  std::vector<uint16_t> col; std::vector<float> del;
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < G; ++g) {
      uint8_t* d = P.data.data() + ((size_t)n * G + g) * 44; uint32_t pl[3] = {0, 0, 0};
      for (int j = 0; j < 32; ++j) {
        uint16_t w = W[(size_t)n * K + g * 32 + j]; int sign = w >> 15, e = (w >> 7) & 0xFF, m = w & 0x7F;
        int idx = -1; for (int i = 0; i < 8; ++i) if (cb8[i] == e) { idx = i; break; }
        if (idx < 0) {  // exception: nearest exponent in codebook
          int best = 0, bd = 1 << 30; for (int i = 0; i < 8; ++i) { int dd = std::abs((int)cb8[i] - e); if (dd < bd) { bd = dd; best = i; } }
          idx = best; uint16_t dec = (uint16_t)((sign << 15) | (cb8[idx] << 7) | m);
          col.push_back((uint16_t)(g * 32 + j)); del.push_back(bf16f(w) - bf16f(dec));
        }
        d[j] = (uint8_t)((sign << 7) | m);
        for (int k = 0; k < 3; ++k) pl[k] |= (uint32_t)((idx >> k) & 1) << j;
      }
      std::memcpy(d + 32, pl, 12);
    }
    P.exc_ptr[n + 1] = (int64_t)col.size();
  }
  P.exc_col = std::move(col); P.exc_delta = std::move(del);
}

static inline __m256i expand_mask(uint32_t m) {  // bit j of m -> byte j = 0xFF/0x00
  __m256i v = _mm256_set1_epi32((int)m);
  const __m256i sh = _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
  v = _mm256_shuffle_epi8(v, sh);
  const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
  return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
}

// decode one 44-byte group into four fp32 vectors for weights [0-7],[16-23],[8-15],[24-31]
static inline void decode_group(const uint8_t* p, __m256i cbv, __m256& a0, __m256& a1, __m256& a2, __m256& a3) {
  __m256i sm = _mm256_loadu_si256((const __m256i*)p);
  uint32_t p0, p1, p2; std::memcpy(&p0, p + 32, 4); std::memcpy(&p1, p + 36, 4); std::memcpy(&p2, p + 40, 4);
  __m256i idx = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(expand_mask(p0), _mm256_set1_epi8(1)),
                                                _mm256_and_si256(expand_mask(p1), _mm256_set1_epi8(2))),
                                _mm256_and_si256(expand_mask(p2), _mm256_set1_epi8(4)));
  __m256i e = _mm256_shuffle_epi8(cbv, idx);                                   // exponent bytes
  __m256i hi = _mm256_or_si256(_mm256_and_si256(sm, _mm256_set1_epi8((char)0x80)),
                               _mm256_and_si256(_mm256_srli_epi16(e, 1), _mm256_set1_epi8(0x7F)));
  __m256i lo = _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi16(e, 7), _mm256_set1_epi8((char)0x80)),
                               _mm256_and_si256(sm, _mm256_set1_epi8(0x7F)));
  __m256i A = _mm256_unpacklo_epi8(lo, hi), B = _mm256_unpackhi_epi8(lo, hi);   // bf16 words
  a0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(A)), 16));
  a1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(A, 1)), 16));
  a2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(B)), 16));
  a3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(B, 1)), 16));
}

static inline float hsum8(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_hadd_ps(a, a); a = _mm_hadd_ps(a, a); return _mm_cvtss_f32(a);
}

template <int MT>
inline void rows_tile(const Packed& P, const float* X, int ldx, int n0, int n1, float* Y, int ldy) {
  const int G = P.K / 32; __m256i cbv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)P.cb));
  for (int n = n0; n < n1; ++n) {
    const uint8_t* row = P.data.data() + (size_t)n * G * 44;
    __m256 acc[MT];
    for (int m = 0; m < MT; ++m) acc[m] = _mm256_setzero_ps();
    for (int g = 0; g < G; ++g) {
      if (g + 12 < G) _mm_prefetch((const char*)(row + (size_t)(g + 12) * 44), _MM_HINT_T0);
      __m256 a0, a1, a2, a3; decode_group(row + (size_t)g * 44, cbv, a0, a1, a2, a3);
      for (int m = 0; m < MT; ++m) {
        const float* x = X + (size_t)m * ldx + g * 32;
        acc[m] = _mm256_fmadd_ps(a0, _mm256_loadu_ps(x), acc[m]);
        acc[m] = _mm256_fmadd_ps(a1, _mm256_loadu_ps(x + 16), acc[m]);
        acc[m] = _mm256_fmadd_ps(a2, _mm256_loadu_ps(x + 8), acc[m]);
        acc[m] = _mm256_fmadd_ps(a3, _mm256_loadu_ps(x + 24), acc[m]);
      }
    }
    for (int m = 0; m < MT; ++m) {
      float y = hsum8(acc[m]);
      for (int64_t e = P.exc_ptr[n]; e < P.exc_ptr[n + 1]; ++e) y += P.exc_delta[e] * X[(size_t)m * ldx + P.exc_col[e]];
      Y[(size_t)m * ldy + n] = y;
    }
  }
}

inline void gemm(const Packed& P, const float* X, int M, float* Y) {   // Y[M,N]
  const int N = P.N, K = P.K; constexpr int CH = 16; const int nch = (N + CH - 1) / CH;
  #pragma omp parallel for schedule(static)
  for (int c = 0; c < nch; ++c) {
    int n0 = c * CH, n1 = std::min(N, n0 + CH), m = 0;
    for (; m + 4 <= M; m += 4) rows_tile<4>(P, X + (size_t)m * K, K, n0, n1, Y + (size_t)m * N, N);
    for (; m < M; ++m) rows_tile<1>(P, X + (size_t)m * K, K, n0, n1, Y + (size_t)m * N, N);
  }
}
}  // namespace bf16p
