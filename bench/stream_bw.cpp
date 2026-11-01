#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <atomic>
static size_t N = 1056ull << 20; static char* buf; static int T; static std::atomic<int> bar{0}; static int cpus[16];
static float sink[16][8];
static void* work(void* a) { int t = (int)(long)a; cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpus[t], &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s);
  size_t c0 = N / T / 64 * 64 * t, c1 = t == T - 1 ? N : N / T / 64 * 64 * (t + 1); __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  for (int rep = 0; rep < 12; ++rep) { for (size_t i = c0; i + 64 <= c1; i += 64) { a0 = _mm256_add_ps(a0, _mm256_load_ps((float*)(buf + i))); a1 = _mm256_add_ps(a1, _mm256_load_ps((float*)(buf + i + 32))); } }
  _mm256_storeu_ps(sink[t], _mm256_add_ps(a0, a1)); return nullptr; }
// usage: stream_bw [threads=8]; threads are pinned to CPUs 0, S, 2S, ... with S = $BW_STRIDE (default 2: one logical CPU per physical core when SMT siblings are numbered 0/1, 2/3, ...)
int main(int argc, char** argv) { T = argc > 1 ? atoi(argv[1]) : 8; const int stride = getenv("BW_STRIDE") ? atoi(getenv("BW_STRIDE")) : 2; for (int i = 0; i < 16; ++i) cpus[i] = i * stride; buf = (char*)aligned_alloc(64, N); memset(buf, 1, N);
  auto t0 = std::chrono::steady_clock::now(); std::vector<pthread_t> th(T); for (long i = 0; i < T; ++i) pthread_create(&th[i], nullptr, work, (void*)i); for (auto& x : th) pthread_join(x, nullptr);
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); printf("T=%d: %.2f GB/s (12 passes x %.0f MB)\n", T, 12.0 * N / s / 1e9, N / 1048576.0); }
