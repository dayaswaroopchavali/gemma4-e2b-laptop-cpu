// Engine v1: low-latency C++ habits applied to the token loop (spec: proofs-cpu/ENGINE_SPEC.md).
//  * no heap allocation and no OpenMP fork/join in the token loop: persistent pinned worker threads,
//    spin-then-park on an atomic generation counter, per-thread cache-line-padded completion flags;
//  * q/k/v and gate/up matmuls batched into ONE parallel phase (fewer synchronisations per layer);
//  * vectorised gelu and bf16 rounding (no libm tanhf in the loop); rope tables precomputed once;
//  * the head matmul fuses the argmax (per-thread max, exact tie rule applied afterwards);
//  * weights pre-faulted (MAP_POPULATE) and locked; one warm-up token before timing.
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sched.h>
#include <pthread.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "w4g32_avx2.h"
#include "bf16_exp3_head.h"
#include "w4a8_avx2.h"
#include <deque>
#include <iostream>

static inline float bf(uint16_t b) { return w4::bf16_to_f32(b); }
static inline uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
static inline float r16(float f) { return bf(f2bf(f)); }
// ---- vector helpers -------------------------------------------------------------------------
static inline __m256 rbf8(__m256 x) {   // round to bf16 (RNE) and back, 8 lanes
  __m256i u = _mm256_castps_si256(x);
  __m256i t = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
  u = _mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7fff)), t);
  return _mm256_castsi256_ps(_mm256_and_si256(u, _mm256_set1_epi32((int)0xFFFF0000u)));
}
static inline __m256 exp8(__m256 x) {   // Cephes-style expf, rel. error ~1e-7
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.3f)), _mm256_set1_ps(88.3f));
  __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f)));
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x); x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 z = _mm256_mul_ps(x, x), y = _mm256_set1_ps(1.9875691500E-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f)); y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f)); y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f)); y = _mm256_fmadd_ps(y, z, _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
  __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}
static inline __m256 gelu8(__m256 x) {  // 0.5 x (1 + tanh(c (x + 0.044715 x^3))) with tanh(u) = 1 - 2/(exp(2u)+1)
  __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
  __m256 u = _mm256_mul_ps(_mm256_set1_ps(0.7978845608028654f), _mm256_fmadd_ps(_mm256_set1_ps(0.044715f), x3, x));
  __m256 e = exp8(_mm256_mul_ps(_mm256_set1_ps(2.0f), u));
  __m256 th = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_div_ps(_mm256_set1_ps(2.0f), _mm256_add_ps(e, _mm256_set1_ps(1.0f))));
  return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), _mm256_add_ps(_mm256_set1_ps(1.0f), th));
}
static inline float gelu_s(float x) { const float c = 0.7978845608028654f; return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x))); }

// ---- persistent pinned thread pool (spin, then park) ----------------------------------------
#include <linux/futex.h>
#include <sys/syscall.h>
#ifdef PROF_NM
#include <x86intrin.h>
struct ProfE { void* f; unsigned long long own, wait, n; }; static ProfE PROFV[32]; static int NPROF = 0;
static void prof_add(void* f, unsigned long long own, unsigned long long wait) { int i = 0; for (; i < NPROF; ++i) if (PROFV[i].f == f) break; if (i == NPROF) { PROFV[i].f = f; ++NPROF; } PROFV[i].own += own; PROFV[i].wait += wait; ++PROFV[i].n; }
#endif
static inline uint64_t now_ns() { return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const uint64_t PARK_NS = (getenv("ENGINE_PARK_US") ? (uint64_t)atoll(getenv("ENGINE_PARK_US")) : 2000ull) * 1000ull;   // spin budget before an idle worker sleeps
struct Pool {
  int T = 1; std::atomic<int> activeT{0}; alignas(64) std::atomic<uint64_t> gen{0};
  std::atomic<int> cur_active{0};
  struct alignas(64) Done { std::atomic<uint64_t> v{0}; };
  struct alignas(64) Enable { std::atomic<uint32_t> v{1}; };
  Done done[16]; Enable en[16]; void (*fn)(void*, int, int) = nullptr; void* arg = nullptr; std::atomic<bool> stop{false}; std::vector<std::thread> th;
  static void pin(int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof(s), &s); }
  void start(const std::vector<int>& cpus) {
    T = (int)cpus.size(); activeT.store(T, std::memory_order_relaxed); cur_active.store(T, std::memory_order_relaxed); pin(cpus[0]);
    for (int t = 1; t < T; ++t) th.emplace_back([this, t, c = cpus[t]] { pin(c); uint64_t last = 0;   // `last` = last job sequence number this worker has looked at
      uint64_t idle0 = now_ns(); unsigned spins = 0;
      for (;;) {
        uint64_t w;   // job word = (sequence << 8) | active-thread-count, published atomically: the activity decision for a job travels WITH the job
        while (((w = gen.load(std::memory_order_acquire)) >> 8) == last) {
          if (stop.load(std::memory_order_relaxed)) return;
          if (__builtin_expect(t >= activeT.load(std::memory_order_relaxed), 0)) {
            en[t].v.store(0, std::memory_order_seq_cst);
            if (t >= activeT.load(std::memory_order_seq_cst) && !stop.load(std::memory_order_relaxed) && (gen.load(std::memory_order_acquire) >> 8) == last) {
              syscall(SYS_futex, reinterpret_cast<uint32_t*>(&en[t].v), FUTEX_WAIT_PRIVATE, 0, nullptr, nullptr, 0);
            }
            en[t].v.store(1, std::memory_order_relaxed);
            continue;   // NOTE: do not touch `last` here: a job published while we were parked must still be examined below
          }
          if (__builtin_expect((++spins & 255) == 0, 0) && now_ns() - idle0 > PARK_NS) {   // idle longer than any gap inside a request: sleep until the next dispatch
            en[t].v.store(0, std::memory_order_seq_cst);
            if (!stop.load(std::memory_order_seq_cst) && (gen.load(std::memory_order_seq_cst) >> 8) == last && t < activeT.load(std::memory_order_seq_cst))
              syscall(SYS_futex, reinterpret_cast<uint32_t*>(&en[t].v), FUTEX_WAIT_PRIVATE, 0, nullptr, nullptr, 0);
            en[t].v.store(1, std::memory_order_relaxed); idle0 = now_ns(); continue;
          }
          _mm_pause();
        }
        last = w >> 8; if (stop.load(std::memory_order_relaxed)) return;
        const int a = (int)(w & 0xFF);
        if (t < a) { fn(arg, t, a); done[t].v.store(last, std::memory_order_release); }
        idle0 = now_ns();
      } });
  }
  void set_active(int a) {
    int target = a < 1 ? T : (a > T ? T : a);
    int prev = activeT.exchange(target, std::memory_order_acq_rel);
    if (target > prev) {
      for (int t = prev; t < target; ++t) {
        en[t].v.store(1, std::memory_order_release);
        syscall(SYS_futex, reinterpret_cast<uint32_t*>(&en[t].v), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
      }
    }
  }
  int active_count() const { return activeT.load(std::memory_order_relaxed); }
  void run(void (*f)(void*, int, int), void* a) {
    fn = f; arg = a; const int active = active_count();
    const uint64_t seq = (gen.load(std::memory_order_relaxed) >> 8) + 1;
    gen.store((seq << 8) | (uint64_t)active, std::memory_order_seq_cst);
    for (int t = 1; t < active; ++t) if (__builtin_expect(en[t].v.load(std::memory_order_seq_cst) == 0, 0)) { en[t].v.store(1, std::memory_order_seq_cst); syscall(SYS_futex, reinterpret_cast<uint32_t*>(&en[t].v), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0); }
#ifdef PROF_NM
    unsigned long long c0 = __rdtsc(), c1;
    f(a, 0, active); c1 = __rdtsc();
    for (int t = 1; t < active; ++t) while (done[t].v.load(std::memory_order_acquire) != seq) _mm_pause();
    unsigned long long c2 = __rdtsc(); prof_add((void*)f, c1 - c0, c2 - c1);
#else
    f(a, 0, active); for (int t = 1; t < active; ++t) while (done[t].v.load(std::memory_order_acquire) != seq) _mm_pause();
#endif
  }
  void shutdown() {
    stop.store(true, std::memory_order_seq_cst);
    gen.fetch_add(1ull << 8, std::memory_order_release);
    for (int t = 1; t < T; ++t) { en[t].v.store(1, std::memory_order_release); syscall(SYS_futex, reinterpret_cast<uint32_t*>(&en[t].v), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0); }
    for (auto& t : th) t.join();
  }
};
static Pool POOL;
template <class F> struct PForCtx { F* f; int n; static void tramp(void* p, int tid, int T) { auto* c = (PForCtx*)p; int lo = (int)((long)tid * c->n / T), hi = (int)((long)(tid + 1) * c->n / T); for (int i = lo; i < hi; ++i) (*c->f)(i, tid); } };
template <class F> static void pfor(int n, F&& f) { PForCtx<std::remove_reference_t<F>> c{&f, n}; POOL.run(PForCtx<std::remove_reference_t<F>>::tramp, &c); }
alignas(64) static float TB[16][512], TMP[16][1536], SC[16][40960], PJ[16][35 * 256];

struct TInfo { std::string dtype; size_t off, nbytes; std::vector<int64_t> shape; };
static std::map<std::string, TInfo> IDX; static uint8_t* BASE;
// ---- residency for a long-lived server: anonymous memory (repacked weights, head screen, KV) can only be swapped, so lock it (as far as RLIMIT_MEMLOCK allows);
// clean file-backed weight pages can be evicted and re-read, so a low-priority thread re-faults any evicted hot page while the engine is idle.
static std::atomic<long long> g_last_req_ns{0};
static void lock_hot_anon() {
  FILE* f = fopen("/proc/self/maps", "r"); if (!f) return; std::vector<std::pair<size_t, uintptr_t>> rs; char line[512];
  while (fgets(line, sizeof line, f)) { unsigned long a, b; char perms[8]; char path[256] = ""; int n = sscanf(line, "%lx-%lx %7s %*s %*s %*s %255s", &a, &b, perms, path);
    if (n >= 3 && perms[0] == 'r' && perms[1] == 'w' && perms[3] == 'p' && (n == 3 || path[0] == '[') && (b - a) >= (8u << 20)) rs.push_back({(size_t)(b - a), (uintptr_t)a}); }
  fclose(f); std::sort(rs.rbegin(), rs.rend()); size_t locked = 0, total = 0;
  for (auto& r : rs) { total += r.first; if (mlock((void*)r.second, r.first) == 0) locked += r.first; }
  fprintf(stderr, "mlock: locked %.0f of %.0f MB anonymous memory (RLIMIT_MEMLOCK permitting)\n", locked / 1e6, total / 1e6);
}
static void keepwarm_loop(std::vector<std::pair<const uint8_t*, size_t>> ranges, std::vector<int> avoid_cpus) {
  cpu_set_t set; CPU_ZERO(&set); const int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN); for (int c = 0; c < ncpu; ++c) if (std::find(avoid_cpus.begin(), avoid_cpus.end(), c) == avoid_cpus.end()) CPU_SET(c, &set);
  if (CPU_COUNT(&set) > 0) pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  const size_t pg = (size_t)sysconf(_SC_PAGESIZE); long long healed = 0;
  for (;;) { std::this_thread::sleep_for(std::chrono::seconds(5));
    if ((long long)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - g_last_req_ns.load() < 2000000000LL) continue;   // only while idle
    long long n0 = healed;
    for (auto& r : ranges) { uintptr_t a = (uintptr_t)r.first & ~(pg - 1); size_t len = (((uintptr_t)r.first + r.second) - a + pg - 1) & ~(pg - 1); std::vector<unsigned char> vec(len / pg);
      if (mincore((void*)a, len, vec.data()) != 0) continue; for (size_t i = 0; i < vec.size(); ++i) if (!(vec[i] & 1)) { volatile uint8_t sink = *(volatile const uint8_t*)(a + i * pg); (void)sink; ++healed; } }
    if (healed != n0) fprintf(stderr, "keepwarm: re-faulted %lld evicted weight pages\n", healed - n0); } }
static const TInfo& ti(const std::string& n) { auto it = IDX.find("model.language_model." + n); if (it == IDX.end()) it = IDX.find(n); if (it == IDX.end()) { fprintf(stderr, "missing tensor %s\n", n.c_str()); exit(1); } return it->second; }
static const uint16_t* bfp(const std::string& n) { return (const uint16_t*)(BASE + ti(n).off); }
static std::vector<float> bfvec(const std::string& n) { auto& t = ti(n); std::vector<float> v(t.nbytes / 2); const uint16_t* p = (const uint16_t*)(BASE + t.off); for (size_t i = 0; i < v.size(); ++i) v[i] = bf(p[i]); return v; }
static bool G_I8 = false, G_I8_ACTIVE = false; static std::deque<w4a8::Packed> PK8;
struct Q4 { const uint8_t* w; const uint16_t* s; int N, K; const w4a8::Packed* p8 = nullptr; };
static Q4 q4(const std::string& base) { auto& wp = ti(base + ".weight_packed"); auto& sc = ti(base + ".weight_scale"); Q4 q; q.w = BASE + wp.off; q.s = (const uint16_t*)(BASE + sc.off); q.N = (int)wp.shape[0]; q.K = (int)wp.shape[1] * 8; if (G_I8) { PK8.emplace_back(); w4a8::repack_alloc(q.N, q.K, PK8.back()); q.p8 = &PK8.back(); } return q; }


constexpr int MM = 32;   // max rows per pass (prefill block; verify uses <= 4)
// activation prep for up to MM rows: xe/xo/gs laid out [m][..]
struct XP { alignas(64) float xe[MM * 6144], xo[MM * 6144], gs[MM * 384]; int K = 0, M = 0; const float* src = nullptr; bool qdone = false; w4a8::XQ xq;
  void prep(const float* x, int K_, int M_) { K = K_; M = M_; src = x; qdone = false;
    const __m256i perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    for (int m = 0; m < M_; ++m) {
      const float* xm = x + (size_t)m * K_; float* xem = xe + (size_t)m * (K_ / 2); float* xom = xo + (size_t)m * (K_ / 2); float* gsm = gs + (size_t)m * (K_ / 32);
      for (int g = 0; g < K_ / 32; ++g) {
        const float* xg = xm + g * 32;
        __m256 v0 = _mm256_loadu_ps(xg + 0), v1 = _mm256_loadu_ps(xg + 8), v2 = _mm256_loadu_ps(xg + 16), v3 = _mm256_loadu_ps(xg + 24);
        __m256 p0 = _mm256_permutevar8x32_ps(v0, perm), p1 = _mm256_permutevar8x32_ps(v1, perm), p2 = _mm256_permutevar8x32_ps(v2, perm), p3 = _mm256_permutevar8x32_ps(v3, perm);
        _mm256_storeu_ps(xem + g * 16 + 0, _mm256_permute2f128_ps(p0, p1, 0x20)); _mm256_storeu_ps(xom + g * 16 + 0, _mm256_permute2f128_ps(p0, p1, 0x31));
        _mm256_storeu_ps(xem + g * 16 + 8, _mm256_permute2f128_ps(p2, p3, 0x20)); _mm256_storeu_ps(xom + g * 16 + 8, _mm256_permute2f128_ps(p2, p3, 0x31));
        float s = 0; for (int i = 0; i < 32; ++i) s += xg[i]; gsm[g] = s;
      }
    }
  } };
static inline void prep_g32_from_vecs(__m256 v0, __m256 v1, __m256 v2, __m256 v3, float* __restrict__ xe, float* __restrict__ xo, float* __restrict__ gs) {
  const __m256i perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  __m256 p0 = _mm256_permutevar8x32_ps(v0, perm), p1 = _mm256_permutevar8x32_ps(v1, perm);
  __m256 p2 = _mm256_permutevar8x32_ps(v2, perm), p3 = _mm256_permutevar8x32_ps(v3, perm);
  _mm256_storeu_ps(xe + 0, _mm256_permute2f128_ps(p0, p1, 0x20));
  _mm256_storeu_ps(xo + 0, _mm256_permute2f128_ps(p0, p1, 0x31));
  _mm256_storeu_ps(xe + 8, _mm256_permute2f128_ps(p2, p3, 0x20));
  _mm256_storeu_ps(xo + 8, _mm256_permute2f128_ps(p2, p3, 0x31));
  alignas(32) float tmp[32];
  _mm256_store_ps(tmp + 0, v0); _mm256_store_ps(tmp + 8, v1);
  _mm256_store_ps(tmp + 16, v2); _mm256_store_ps(tmp + 24, v3);
  float s = 0; for (int i = 0; i < 32; ++i) s += tmp[i];
  *gs = s;
}
static inline void prep_x_avx2(const float* __restrict__ x, int K, float* __restrict__ xe, float* __restrict__ xo, float* __restrict__ gs) {
  const __m256i perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  for (int g = 0; g < K / 32; ++g) {
    const float* xg = x + g * 32;
    __m256 v0 = _mm256_loadu_ps(xg + 0),  v1 = _mm256_loadu_ps(xg + 8);
    __m256 v2 = _mm256_loadu_ps(xg + 16), v3 = _mm256_loadu_ps(xg + 24);
    __m256 p0 = _mm256_permutevar8x32_ps(v0, perm), p1 = _mm256_permutevar8x32_ps(v1, perm);
    __m256 p2 = _mm256_permutevar8x32_ps(v2, perm), p3 = _mm256_permutevar8x32_ps(v3, perm);
    _mm256_storeu_ps(xe + g * 16 + 0, _mm256_permute2f128_ps(p0, p1, 0x20));
    _mm256_storeu_ps(xo + g * 16 + 0, _mm256_permute2f128_ps(p0, p1, 0x31));
    _mm256_storeu_ps(xe + g * 16 + 8, _mm256_permute2f128_ps(p2, p3, 0x20));
    _mm256_storeu_ps(xo + g * 16 + 8, _mm256_permute2f128_ps(p2, p3, 0x31));
    float s = 0; for (int i = 0; i < 32; ++i) s += xg[i];
    gs[g] = s;
  }
}
// Static 7/8 + dynamic 1/8 work split: each thread keeps a contiguous block (streaming locality), the last 1/8 of every block is handed out in
// small chunks via an atomic counter, so a slow thread cannot stall the join. Units are independent output rows, so results are bit-identical.
#ifndef DYN_NUM
#define DYN_NUM 7
#endif
#ifndef DYN_CH
#define DYN_CH 4
#endif
template <class F> static inline void split_dyn(std::atomic<int>& nxt, int total, int tid, int T, F&& run) {
  const long b0 = (long)tid * total / T, b1 = (long)(tid + 1) * total / T; run(b0, b0 + (b1 - b0) * DYN_NUM / 8);
  const int nck = (int)((((total + T - 1) / T) - ((total / T) * DYN_NUM / 8) + DYN_CH - 1) / DYN_CH);
  for (int g; (g = nxt.fetch_add(1, std::memory_order_relaxed)) < T * nck;) {
    const int i = g / nck, k = g % nck; const long a0 = (long)i * total / T, a1 = (long)(i + 1) * total / T; const long lo = a0 + (a1 - a0) * DYN_NUM / 8 + (long)k * DYN_CH, hi = std::min<long>(lo + DYN_CH, a1);
    if (lo < hi) run(lo, hi);
  }
}
struct LinArg { const XP* x; const Q4* q[4]; float* y[4]; int nj; int cum[5]; std::atomic<int> nxt{0}; };
template <int MT> static inline void tile_call(const Q4& q, const XP& x, int n0, int n1, float* y) { w4::gemm_rows_tile<MT>(q.w, q.s, x.xe, x.xo, x.gs, q.K, n0, n1, y, q.N, q.K / 2, q.K / 32); }
static void lin_task(void* p, int tid, int T) {
  LinArg& a = *(LinArg*)p; const int total = a.cum[a.nj];
  split_dyn(a.nxt, total, tid, T, [&](long c0, long c1) {
  for (int j = 0; j < a.nj; ++j) { long lo = std::max<long>(c0, a.cum[j]), hi = std::min<long>(c1, a.cum[j + 1]); if (lo >= hi) continue;
    const Q4& q = *a.q[j]; int n0 = (int)((lo - a.cum[j]) * 32), n1 = (int)std::min<long>(q.N, (hi - a.cum[j]) * 32);
    const int M = a.x->M; const XP& X = *a.x; int m0 = 0;
    for (; m0 + 4 <= M; m0 += 4) w4::gemm_rows_tile<4>(q.w, q.s, X.xe + (size_t)m0 * (q.K / 2), X.xo + (size_t)m0 * (q.K / 2), X.gs + (size_t)m0 * (q.K / 32), q.K, n0, n1, a.y[j] + (size_t)m0 * q.N, q.N, q.K / 2, q.K / 32);
    {
#define REM_CASE(R) case R: w4::gemm_rows_tile<R>(q.w, q.s, X.xe + (size_t)m0 * (q.K / 2), X.xo + (size_t)m0 * (q.K / 2), X.gs + (size_t)m0 * (q.K / 32), q.K, n0, n1, a.y[j] + (size_t)m0 * q.N, q.N, q.K / 2, q.K / 32); break;
      switch (M - m0) { REM_CASE(1) REM_CASE(2) REM_CASE(3) default: break; }
#undef REM_CASE
    } } });
}
struct GateUpArg { const XP* x; const Q4* gate; const Q4* up; float* gbuf; float* ubuf; float* abuf; XP* xb; };
static void gate_up_fused_task(void* p, int tid, int T) {
  GateUpArg& a = *(GateUpArg*)p;
  const int I = a.gate->N, K = a.gate->K, M = a.x->M;
  const int nch = (I + 31) / 32;
  const int c0 = (int)((long)tid * nch / T), c1 = (int)((long)(tid + 1) * nch / T);
  const int n0 = c0 * 32, n1 = std::min(I, c1 * 32);
  if (n0 >= n1) return;
  const XP& X = *a.x;
  int m0 = 0;
  for (; m0 + 4 <= M; m0 += 4) {
    w4::gemm_rows_tile<4>(a.gate->w, a.gate->s, X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.gbuf + (size_t)m0 * I, I, K / 2, K / 32);
    w4::gemm_rows_tile<4>(a.up->w,   a.up->s,   X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.ubuf + (size_t)m0 * I, I, K / 2, K / 32);
  }
  switch (M - m0) {
    case 1:
      w4::gemm_rows_tile<1>(a.gate->w, a.gate->s, X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.gbuf + (size_t)m0 * I, I, K / 2, K / 32);
      w4::gemm_rows_tile<1>(a.up->w,   a.up->s,   X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.ubuf + (size_t)m0 * I, I, K / 2, K / 32);
      break;
    case 2:
      w4::gemm_rows_tile<2>(a.gate->w, a.gate->s, X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.gbuf + (size_t)m0 * I, I, K / 2, K / 32);
      w4::gemm_rows_tile<2>(a.up->w,   a.up->s,   X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.ubuf + (size_t)m0 * I, I, K / 2, K / 32);
      break;
    case 3:
      w4::gemm_rows_tile<3>(a.gate->w, a.gate->s, X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.gbuf + (size_t)m0 * I, I, K / 2, K / 32);
      w4::gemm_rows_tile<3>(a.up->w,   a.up->s,   X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.ubuf + (size_t)m0 * I, I, K / 2, K / 32);
      break;
    default: break;
  }
  // Fused gelu8 + XB.prep on [n0, n1) (which is a multiple of 32)!
  for (int m = 0; m < M; ++m) {
    const float* gm = a.gbuf + (size_t)m * I;
    const float* um = a.ubuf + (size_t)m * I;
    float* am = a.abuf + (size_t)m * I;
    float* xe = a.xb->xe + (size_t)m * (I / 2);
    float* xo = a.xb->xo + (size_t)m * (I / 2);
    float* gs = a.xb->gs + (size_t)m * (I / 32);
    for (int g = n0 / 32; g < n1 / 32; ++g) {
      const int base = g * 32;
      __m256 v0 = rbf8(_mm256_mul_ps(rbf8(gelu8(rbf8(_mm256_loadu_ps(gm + base + 0)))),  rbf8(_mm256_loadu_ps(um + base + 0))));
      __m256 v1 = rbf8(_mm256_mul_ps(rbf8(gelu8(rbf8(_mm256_loadu_ps(gm + base + 8)))),  rbf8(_mm256_loadu_ps(um + base + 8))));
      __m256 v2 = rbf8(_mm256_mul_ps(rbf8(gelu8(rbf8(_mm256_loadu_ps(gm + base + 16)))), rbf8(_mm256_loadu_ps(um + base + 16))));
      __m256 v3 = rbf8(_mm256_mul_ps(rbf8(gelu8(rbf8(_mm256_loadu_ps(gm + base + 24)))), rbf8(_mm256_loadu_ps(um + base + 24))));
      prep_g32_from_vecs(v0, v1, v2, v3, xe + g * 16, xo + g * 16, gs + g);
    }
  }
}
struct I8Arg { const XP* x; const Q4* q[4]; float* y[4]; int nj; int cum[5]; std::atomic<int> nxt{0}; };
static void i8_task(void* p, int tid, int T) {
  I8Arg& a = *(I8Arg*)p; const int total = a.cum[a.nj]; const int M = a.x->M;
  split_dyn(a.nxt, total, tid, T, [&](long c0, long c1) {
  for (int j = 0; j < a.nj; ++j) { long lo = std::max<long>(c0, a.cum[j]), hi = std::min<long>(c1, a.cum[j + 1]); const Q4& q = *a.q[j];
    for (long t = lo; t < hi; ++t) { int tt = (int)(t - a.cum[j]), m = 0; for (; m + 4 <= M; m += 4) w4a8::tile<4>(*q.p8, tt, a.x->xq, m, a.y[j]); for (; m < M; ++m) w4a8::tile<1>(*q.p8, tt, a.x->xq, m, a.y[j]); } } });
}
static void lin_multi(const XP& x0, int nj, const Q4* const* q, float* const* y) {
  if (G_I8_ACTIVE && x0.M >= 2 && q[0]->p8) { XP& x = const_cast<XP&>(x0); if (!x.qdone) { w4a8::quant(x.src, x.M, x.K, x.xq); x.qdone = true; }
    I8Arg a; a.x = &x; a.nj = nj; a.cum[0] = 0; for (int j = 0; j < nj; ++j) { a.q[j] = q[j]; a.y[j] = y[j]; a.cum[j + 1] = a.cum[j] + q[j]->N / 8; } POOL.run(i8_task, &a); return; }
  const XP& x = x0;
  LinArg a; a.x = &x; a.nj = nj; a.cum[0] = 0; for (int j = 0; j < nj; ++j) { a.q[j] = q[j]; a.y[j] = y[j]; a.cum[j + 1] = a.cum[j] + (q[j]->N + 31) / 32; }
  POOL.run(lin_task, &a);
}
static void lin1(const XP& x, const Q4& q, float* y) { const Q4* qq[1] = {&q}; float* yy[1] = {y}; lin_multi(x, 1, qq, yy); }
static inline __m128i f2bf8(__m256 v) {
  __m256i u = _mm256_castps_si256(v);
  __m256i t = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
  u = _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(u, _mm256_set1_epi32(0x7fff)), t), 16);
  __m128i lo = _mm256_castsi256_si128(u);
  __m128i hi = _mm256_extracti128_si256(u, 1);
  return _mm_packus_epi32(lo, hi);
}
static inline double ss_avx2(const float* x, int n) {
  __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
  __m256d s2 = _mm256_setzero_pd(), s3 = _mm256_setzero_pd();
  for (int i = 0; i < n; i += 16) {
    __m256d d0 = _mm256_cvtps_pd(_mm_loadu_ps(x + i + 0));
    __m256d d1 = _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4));
    __m256d d2 = _mm256_cvtps_pd(_mm_loadu_ps(x + i + 8));
    __m256d d3 = _mm256_cvtps_pd(_mm_loadu_ps(x + i + 12));
    s0 = _mm256_fmadd_pd(d0, d0, s0);
    s1 = _mm256_fmadd_pd(d1, d1, s1);
    s2 = _mm256_fmadd_pd(d2, d2, s2);
    s3 = _mm256_fmadd_pd(d3, d3, s3);
  }
  s0 = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));
  __m128d lo = _mm256_castpd256_pd128(s0), hi = _mm256_extractf128_pd(s0, 1);
  lo = _mm_add_pd(lo, hi);
  return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));
}
static inline void rms(const float* x, const float* w, int n, float* y, float eps = 1e-6f) {
  double ss = ss_avx2(x, n); float r = 1.0f / sqrtf((float)(ss / n) + eps);
  __m256 vr = _mm256_set1_ps(r);
  if (w) {
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), vr), _mm256_loadu_ps(w + i)));
  } else {
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vr));
  }
}
static inline void rms_r16_inplace(float* x, const float* w, int n, float eps = 1e-6f) {
  double ss = ss_avx2(x, n); float r = 1.0f / sqrtf((float)(ss / n) + eps);
  __m256 vr = _mm256_set1_ps(r);
  if (w) {
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(x + i, rbf8(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), vr), _mm256_loadu_ps(w + i))));
  } else {
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(x + i, rbf8(_mm256_mul_ps(_mm256_loadu_ps(x + i), vr)));
  }
}
static inline void add_res_inplace(float* x, const float* y, int n) {
  for (int i = 0; i < n; i += 8) _mm256_storeu_ps(x + i, rbf8(_mm256_add_ps(_mm256_loadu_ps(x + i), rbf8(_mm256_loadu_ps(y + i)))));
}
static inline void rms_prep_x(const float* x, const float* w, int n, float* y, float* xe, float* xo, float* gs, bool do_r16) {
  double ss = ss_avx2(x, n); float r = 1.0f / sqrtf((float)(ss / n) + 1e-6f);
  __m256 vr = _mm256_set1_ps(r);
  for (int g = 0; g < n / 32; ++g) {
    const int i = g * 32;
    __m256 v0 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 0),  vr), _mm256_loadu_ps(w + i + 0));
    __m256 v1 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 8),  vr), _mm256_loadu_ps(w + i + 8));
    __m256 v2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 16), vr), _mm256_loadu_ps(w + i + 16));
    __m256 v3 = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i + 24), vr), _mm256_loadu_ps(w + i + 24));
    if (do_r16) { v0 = rbf8(v0); v1 = rbf8(v1); v2 = rbf8(v2); v3 = rbf8(v3); }
    if (y) {
      _mm256_storeu_ps(y + i + 0,  v0); _mm256_storeu_ps(y + i + 8,  v1);
      _mm256_storeu_ps(y + i + 16, v2); _mm256_storeu_ps(y + i + 24, v3);
    }
    prep_g32_from_vecs(v0, v1, v2, v3, xe + g * 16, xo + g * 16, gs + g);
  }
}
static inline void add_res_prep_x(float* x, const float* y, int n, float* xe, float* xo, float* gs) {
  for (int g = 0; g < n / 32; ++g) {
    const int i = g * 32;
    __m256 v0 = rbf8(_mm256_add_ps(_mm256_loadu_ps(x + i + 0),  rbf8(_mm256_loadu_ps(y + i + 0))));
    __m256 v1 = rbf8(_mm256_add_ps(_mm256_loadu_ps(x + i + 8),  rbf8(_mm256_loadu_ps(y + i + 8))));
    __m256 v2 = rbf8(_mm256_add_ps(_mm256_loadu_ps(x + i + 16), rbf8(_mm256_loadu_ps(y + i + 16))));
    __m256 v3 = rbf8(_mm256_add_ps(_mm256_loadu_ps(x + i + 24), rbf8(_mm256_loadu_ps(y + i + 24))));
    _mm256_storeu_ps(x + i + 0,  v0); _mm256_storeu_ps(x + i + 8,  v1);
    _mm256_storeu_ps(x + i + 16, v2); _mm256_storeu_ps(x + i + 24, v3);
    prep_g32_from_vecs(v0, v1, v2, v3, xe + g * 16, xo + g * 16, gs + g);
  }
}
static inline void rope_avx2(float* __restrict__ h, const float* __restrict__ cs, const float* __restrict__ sn, int hd) {
  const int hh = hd / 2;
  for (int j = 0; j < hh; j += 8) {
    __m256 u = _mm256_loadu_ps(h + j), w = _mm256_loadu_ps(h + j + hh);
    __m256 c = _mm256_loadu_ps(cs + j), s = _mm256_loadu_ps(sn + j);
    _mm256_storeu_ps(h + j,      _mm256_fmsub_ps(u, c, _mm256_mul_ps(w, s)));
    _mm256_storeu_ps(h + j + hh, _mm256_fmadd_ps(w, c, _mm256_mul_ps(u, s)));
  }
}
static void rbf_n(float* p, int n) { for (int i = 0; i < n; i += 8) _mm256_storeu_ps(p + i, rbf8(_mm256_loadu_ps(p + i))); }
static void add_res(float* x, const float* t, const float* r, int n) { for (int i = 0; i < n; i += 8) _mm256_storeu_ps(x + i, rbf8(_mm256_add_ps(rbf8(_mm256_loadu_ps(t + i)), _mm256_loadu_ps(r + i)))); }
struct PgFusedArg { const XP* x; const Q4* pg; float* pgv; float* g2; const float* pli_base; int l; XP* xb; };
static void pg_fused_task(void* p, int tid, int T) {
  PgFusedArg& a = *(PgFusedArg*)p;
  const int PD = 256, K = a.pg->K, M = a.x->M, nch = PD / 32;
  const int c0 = (int)((long)tid * nch / T), c1 = (int)((long)(tid + 1) * nch / T);
  const int n0 = c0 * 32, n1 = c1 * 32;
  if (n0 >= n1) return;
  const XP& X = *a.x;
  int m0 = 0;
  for (; m0 + 4 <= M; m0 += 4) w4::gemm_rows_tile<4>(a.pg->w, a.pg->s, X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.pgv + (size_t)m0 * PD, PD, K / 2, K / 32);
  switch (M - m0) {
    case 1: w4::gemm_rows_tile<1>(a.pg->w, a.pg->s, X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.pgv + (size_t)m0 * PD, PD, K / 2, K / 32); break;
    case 2: w4::gemm_rows_tile<2>(a.pg->w, a.pg->s, X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.pgv + (size_t)m0 * PD, PD, K / 2, K / 32); break;
    case 3: w4::gemm_rows_tile<3>(a.pg->w, a.pg->s, X.xe + (size_t)m0 * (K / 2), X.xo + (size_t)m0 * (K / 2), X.gs + (size_t)m0 * (K / 32), K, n0, n1, a.pgv + (size_t)m0 * PD, PD, K / 2, K / 32); break;
    default: break;
  }
  for (int m = 0; m < M; ++m) {
    const float* pm = a.pgv + (size_t)m * PD;
    const float* plim = a.pli_base + (size_t)m * (35 * 256) + a.l * PD;
    float* gm = a.g2 + (size_t)m * 256;
    float* xe = a.xb->xe + (size_t)m * 128;
    float* xo = a.xb->xo + (size_t)m * 128;
    float* gs = a.xb->gs + (size_t)m * 8;
    for (int g = c0; g < c1; ++g) {
      const int base = g * 32;
      for (int i = base; i < base + 32; i += 8) {
        __m256 gv = rbf8(_mm256_loadu_ps(pm + i));
        _mm256_storeu_ps(gm + i, rbf8(_mm256_mul_ps(rbf8(gelu8(gv)), rbf8(_mm256_loadu_ps(plim + i)))));
      }
      float s = 0;
      for (int j = 0; j < 16; ++j) {
        float ev = gm[base + 2 * j], od = gm[base + 2 * j + 1];
        xe[g * 16 + j] = ev; xo[g * 16 + j] = od;
        s += ev; s += od;
      }
      gs[g] = s;
    }
  }
}

// ---- certified head screen: int4 g32 copy of the head + per-row error coefficient ----
struct Screen { std::vector<uint8_t> w; std::vector<uint16_t> s; std::vector<float> coef; bool on = false; } SCR;
static const int64_t* g_d_order = nullptr;
static float g_gap = 1e30f;
struct ScrArg {
  const Screen* S; const XP* x; const float* hn; float pre_thr[4]; float tmax[16][4];
  alignas(64) std::atomic<float> shared_best[4]; std::atomic<int> nxt{0};
  int t_idx[16][4][16384]; float t_u[16][4][16384]; int t_cnt[16][4]; std::atomic<bool> overflow[4]; int M;
};
static inline float calc_thr_fast(float L) {
  auto cap = [](float s) { return r16(30.0f * tanhf(s / 30.0f)); };
  float Lr = r16(L), cmL = cap(Lr), lo = Lr - 64.0f, hi = Lr;
  for (int it = 0; it < 40; ++it) { float mid = 0.5f * (lo + hi); if (cap(r16(mid)) == cmL) hi = mid; else lo = mid; }
  return hi - 0.25f;
}
static void screen_task(void* p, int tid, int T) {
  ScrArg& a = *(ScrArg*)p; constexpr int CH = 64, N = 262144, K = 1536; const int nch = N / CH; int iter = 0;
  __m256 vmx[4] = {_mm256_set1_ps(-1e30f), _mm256_set1_ps(-1e30f), _mm256_set1_ps(-1e30f), _mm256_set1_ps(-1e30f)};
  __m256 vhn[4], vthr[4], vbest_L[4]; float cur_best[4]; int cnt[4] = {0, 0, 0, 0};
  for (int m = 0; m < a.M; ++m) {
    vhn[m] = _mm256_set1_ps(a.hn[m]);
    cur_best[m] = a.pre_thr[m];
    vbest_L[m] = _mm256_set1_ps(cur_best[m] + 0.05f);
    vthr[m] = _mm256_set1_ps(calc_thr_fast(cur_best[m]) - 1e-4f);
  }
  const uint8_t* W = a.S->w.data(); const uint16_t* Sc = a.S->s.data(); const float* cf = a.S->coef.data();
  alignas(64) float l1_s[4][CH];
  auto do_chunk = [&](long c) {
    if (__builtin_expect((++iter & 7) == 0, 0)) {
      for (int m = 0; m < a.M; ++m) {
        float sb = a.shared_best[m].load(std::memory_order_relaxed);
        if (sb > cur_best[m] + 0.05f) {
          cur_best[m] = sb;
          vbest_L[m] = _mm256_set1_ps(sb + 0.05f);
          vthr[m] = _mm256_set1_ps(calc_thr_fast(sb) - 1e-4f);
        }
      }
    }
    int n0 = (int)(c * CH), n1 = n0 + CH;
    const uint8_t* Wc = W + (size_t)n0 * (K / 2); const uint16_t* Sc_c = Sc + (size_t)n0 * (K / 32);
    switch (a.M) {
      case 1: w4::gemm_rows_tile<1>(Wc, Sc_c, a.x->xe, a.x->xo, a.x->gs, K, 0, CH, &l1_s[0][0], CH, K / 2, K / 32); break;
      case 2: w4::gemm_rows_tile<2>(Wc, Sc_c, a.x->xe, a.x->xo, a.x->gs, K, 0, CH, &l1_s[0][0], CH, K / 2, K / 32); break;
      case 3: w4::gemm_rows_tile<3>(Wc, Sc_c, a.x->xe, a.x->xo, a.x->gs, K, 0, CH, &l1_s[0][0], CH, K / 2, K / 32); break;
      default: w4::gemm_rows_tile<4>(Wc, Sc_c, a.x->xe, a.x->xo, a.x->gs, K, 0, CH, &l1_s[0][0], CH, K / 2, K / 32); break;
    }
    for (int i = 0; i < CH; i += 8) {
      int n = n0 + i;
      __m256 c8 = _mm256_loadu_ps(cf + n);
      for (int m = 0; m < a.M; ++m) {
        __m256 s8 = _mm256_load_ps(&l1_s[m][i]);
        vmx[m] = _mm256_max_ps(vmx[m], _mm256_fnmadd_ps(c8, vhn[m], s8));
        __m256 u8 = _mm256_fmadd_ps(c8, vhn[m], s8);
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(u8, vthr[m], _CMP_GE_OQ));
        if (__builtin_expect(mask != 0, 0)) {
          alignas(32) float ub[8]; _mm256_store_ps(ub, u8);
          int* idx_out = a.t_idx[tid][m]; float* u_out = a.t_u[tid][m]; int cm = cnt[m];
          while (mask) {
            int bit = __builtin_ctz(mask);
            mask &= mask - 1;
            if (__builtin_expect(cm < 16384, 1)) {
              idx_out[cm] = n + bit;
              u_out[cm] = ub[bit];
              ++cm;
            } else {
              a.overflow[m].store(true, std::memory_order_relaxed);
            }
          }
          cnt[m] = cm;
        }
      }
    }
    for (int m = 0; m < a.M; ++m) {
      if (__builtin_expect(_mm256_movemask_ps(_mm256_cmp_ps(vmx[m], vbest_L[m], _CMP_GT_OQ)) != 0, 0)) {
        float tb[8]; _mm256_storeu_ps(tb, vmx[m]);
        float mx = tb[0]; for (int i = 1; i < 8; ++i) mx = std::max(mx, tb[i]);
        if (mx > cur_best[m] + 0.05f) {
          cur_best[m] = mx;
          vbest_L[m] = _mm256_set1_ps(mx + 0.05f);
          vthr[m] = _mm256_set1_ps(calc_thr_fast(mx) - 1e-4f);
          float prev = a.shared_best[m].load(std::memory_order_relaxed);
          while (mx > prev && !a.shared_best[m].compare_exchange_weak(prev, mx, std::memory_order_relaxed)) {}
        }
      }
    }
  };
  split_dyn(a.nxt, nch, tid, T, [&](long lo, long hi) { for (long c = lo; c < hi; ++c) do_chunk(c); });
  for (int m = 0; m < a.M; ++m) {
    a.t_cnt[tid][m] = cnt[m];
    float tb[8]; _mm256_storeu_ps(tb, vmx[m]);
    float mx = tb[0]; for (int i = 1; i < 8; ++i) mx = std::max(mx, tb[i]);
    a.tmax[tid][m] = mx;
  }
}
struct RescArg { const bf16p::Packed* P; const float* h; float* fval; const int* idx; float tmax[16]; int cnt; };
static void resc_task(void* p, int tid, int T) {
  RescArg& a = *(RescArg*)p; long c0 = (long)tid * a.cnt / T, c1 = (long)(tid + 1) * a.cnt / T;
  const bf16p::Packed& P = *a.P; const float* X = a.h;
  const int G = P.K / 32; __m256i cbv = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)P.cb));
  float mx = -1e30f;
  for (long i = c0; i < c1; ++i) {
    int n = a.idx[i];
    if (i + 1 < c1) {
      int nn = a.idx[i + 1];
      const uint8_t* nrow = P.data.data() + (size_t)nn * G * 44;
      for (int cl = 0; cl < 12; ++cl) _mm_prefetch((const char*)(nrow + cl * 64), _MM_HINT_T0);
      _mm_prefetch((const char*)(&P.exc_ptr[nn]), _MM_HINT_T0);
    }
    const uint8_t* row = P.data.data() + (size_t)n * G * 44;
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < G; ++g) {
      if (g + 12 < G) _mm_prefetch((const char*)(row + (size_t)(g + 12) * 44), _MM_HINT_T0);
      __m256 a0, a1, a2, a3; bf16p::decode_group(row + (size_t)g * 44, cbv, a0, a1, a2, a3);
      const float* x = X + g * 32;
      acc = _mm256_fmadd_ps(a0, _mm256_loadu_ps(x), acc);
      acc = _mm256_fmadd_ps(a1, _mm256_loadu_ps(x + 16), acc);
      acc = _mm256_fmadd_ps(a2, _mm256_loadu_ps(x + 8), acc);
      acc = _mm256_fmadd_ps(a3, _mm256_loadu_ps(x + 24), acc);
    }
    float y = bf16p::hsum8(acc);
    for (int64_t e = P.exc_ptr[n]; e < P.exc_ptr[n + 1]; ++e) y += P.exc_delta[e] * X[P.exc_col[e]];
    float v = r16(y);
    a.fval[i] = v;
    mx = std::max(mx, v);
  }
  a.tmax[tid] = mx;
}
struct Layer { Q4 qp, kp, vp, op, gate, up, down, pg, pp; bool owner; std::vector<float> in_ln, post_attn, pre_ff, post_ff, post_ple, qn, kn; float scalar; int hd; bool sliding; };
struct HeadArg { const bf16p::Packed* P; const float* h; float* logit; int M; };
static void head_task(void* p, int tid, int T) {
  HeadArg& a = *(HeadArg*)p; constexpr int CH = 16; const int nch = (a.P->N + CH - 1) / CH; long c0 = (long)tid * nch / T, c1 = (long)(tid + 1) * nch / T;
  for (long c = c0; c < c1; ++c) { int n0 = (int)(c * CH), n1 = std::min(a.P->N, n0 + CH); const int K = a.P->K, N = a.P->N;
    switch (a.M) { case 1: bf16p::rows_tile<1>(*a.P, a.h, K, n0, n1, a.logit, N); break; case 2: bf16p::rows_tile<2>(*a.P, a.h, K, n0, n1, a.logit, N); break;
      case 3: bf16p::rows_tile<3>(*a.P, a.h, K, n0, n1, a.logit, N); break; default: bf16p::rows_tile<4>(*a.P, a.h, K, n0, n1, a.logit, N); } }
}


// ---------------- drafter (Google gemma-4-E2B-it-assistant): bf16 weights ----------------
struct B16 { const uint16_t* w; int N, K; };
struct BArg { const B16* m[2]; float* y[2]; const float* x; int nj; int cum[3]; };
static inline __m256 bf8(const uint16_t* p) { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)p)), 16)); }
static void b16_rows(const B16& m, const float* x, int n0, int n1, float* y) {
  int n = n0;
  for (; n + 4 <= n1; n += 4) {
    const uint16_t* w0 = m.w + (size_t)(n + 0) * m.K;
    const uint16_t* w1 = m.w + (size_t)(n + 1) * m.K;
    const uint16_t* w2 = m.w + (size_t)(n + 2) * m.K;
    const uint16_t* w3 = m.w + (size_t)(n + 3) * m.K;
    __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
    for (int k = 0; k < m.K; k += 16) {
      __m256 x0 = _mm256_loadu_ps(x + k), x1 = _mm256_loadu_ps(x + k + 8);
      a0 = _mm256_fmadd_ps(bf8(w0 + k), x0, a0); b0 = _mm256_fmadd_ps(bf8(w0 + k + 8), x1, b0);
      a1 = _mm256_fmadd_ps(bf8(w1 + k), x0, a1); b1 = _mm256_fmadd_ps(bf8(w1 + k + 8), x1, b1);
      a2 = _mm256_fmadd_ps(bf8(w2 + k), x0, a2); b2 = _mm256_fmadd_ps(bf8(w2 + k + 8), x1, b2);
      a3 = _mm256_fmadd_ps(bf8(w3 + k), x0, a3); b3 = _mm256_fmadd_ps(bf8(w3 + k + 8), x1, b3);
    }
    y[n + 0] = w4::hsum(_mm256_add_ps(a0, b0));
    y[n + 1] = w4::hsum(_mm256_add_ps(a1, b1));
    y[n + 2] = w4::hsum(_mm256_add_ps(a2, b2));
    y[n + 3] = w4::hsum(_mm256_add_ps(a3, b3));
  }
  for (; n < n1; ++n) {
    const uint16_t* w = m.w + (size_t)n * m.K; __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int k = 0; k < m.K; k += 16) { a0 = _mm256_fmadd_ps(bf8(w + k), _mm256_loadu_ps(x + k), a0); a1 = _mm256_fmadd_ps(bf8(w + k + 8), _mm256_loadu_ps(x + k + 8), a1); }
    y[n] = w4::hsum(_mm256_add_ps(a0, a1));
  }
}
struct PleFusedArg {
  const XP* x;
  const Q4* plproj;
  const float* plnorm;
  const uint16_t* PLE;
  const int* toks;
  const float* const* emb;
  float* pr;
  float* pli;
  int M, NL, PD, H;
};
static void ple_fused_task(void* p, int tid, int T) {
  PleFusedArg& a = *(PleFusedArg*)p;
  const int l0 = (int)((long)tid * a.NL / T);
  const int l1 = (int)((long)(tid + 1) * a.NL / T);
  if (l0 >= l1) return;
  const int n0 = l0 * a.PD, n1 = l1 * a.PD;
  const Q4& q = *a.plproj;
  const XP& X = *a.x;
  const int M = a.M;
  int m0 = 0;
  for (; m0 + 4 <= M; m0 += 4) w4::gemm_rows_tile<4>(q.w, q.s, X.xe + (size_t)m0 * (q.K / 2), X.xo + (size_t)m0 * (q.K / 2), X.gs + (size_t)m0 * (q.K / 32), q.K, n0, n1, a.pr + (size_t)m0 * q.N, q.N, q.K / 2, q.K / 32);
  switch (M - m0) {
    case 1: w4::gemm_rows_tile<1>(q.w, q.s, X.xe + (size_t)m0 * (q.K / 2), X.xo + (size_t)m0 * (q.K / 2), X.gs + (size_t)m0 * (q.K / 32), q.K, n0, n1, a.pr + (size_t)m0 * q.N, q.N, q.K / 2, q.K / 32); break;
    case 2: w4::gemm_rows_tile<2>(q.w, q.s, X.xe + (size_t)m0 * (q.K / 2), X.xo + (size_t)m0 * (q.K / 2), X.gs + (size_t)m0 * (q.K / 32), q.K, n0, n1, a.pr + (size_t)m0 * q.N, q.N, q.K / 2, q.K / 32); break;
    case 3: w4::gemm_rows_tile<3>(q.w, q.s, X.xe + (size_t)m0 * (q.K / 2), X.xo + (size_t)m0 * (q.K / 2), X.gs + (size_t)m0 * (q.K / 32), q.K, n0, n1, a.pr + (size_t)m0 * q.N, q.N, q.K / 2, q.K / 32); break;
    default: break;
  }
  const float s1 = 1.0f / sqrtf((float)a.H);
  const __m256 vs1 = _mm256_set1_ps(s1), v16 = _mm256_set1_ps(16.0f), vinv2 = _mm256_set1_ps(0.70710678118f);
  for (int m = 0; m < a.M; ++m) {
    const int tok_m = (a.emb && a.emb[m]) ? 0 : a.toks[m];
    for (int l = l0; l < l1; ++l) {
      float* prl = a.pr + (size_t)m * (a.NL * a.PD) + l * a.PD;
      for (int i = 0; i < a.PD; i += 8) _mm256_storeu_ps(prl + i, rbf8(_mm256_mul_ps(_mm256_loadu_ps(prl + i), vs1)));
      alignas(32) float pjl[256];
      rms(prl, a.plnorm, a.PD, pjl);
      const uint16_t* pel = a.PLE + (size_t)tok_m * 8960 + l * a.PD;
      float* plil = a.pli + (size_t)m * (a.NL * a.PD) + l * a.PD;
      for (int i = 0; i < a.PD; i += 8) {
        __m256 pev = rbf8(_mm256_mul_ps(bf8(pel + i), v16));
        _mm256_storeu_ps(plil + i, _mm256_mul_ps(_mm256_add_ps(_mm256_load_ps(pjl + i), pev), vinv2));
      }
    }
  }
}
struct DGateUpArg { const float* x; const B16* gate; const B16* up; float* dgate; float* dup; float* dact; };
static void d_gate_up_task(void* p, int tid, int T) {
  DGateUpArg& a = *(DGateUpArg*)p;
  const int nch = 2048 / 16;
  const int c0 = (int)((long)tid * nch / T), c1 = (int)((long)(tid + 1) * nch / T);
  const int n0 = c0 * 16, n1 = c1 * 16;
  if (n0 >= n1) return;
  b16_rows(*a.gate, a.x, n0, n1, a.dgate);
  b16_rows(*a.up,   a.x, n0, n1, a.dup);
  for (int i = n0; i < n1; i += 8) {
    __m256 g = rbf8(_mm256_loadu_ps(a.dgate + i)), u = rbf8(_mm256_loadu_ps(a.dup + i));
    _mm256_storeu_ps(a.dact + i, rbf8(_mm256_mul_ps(rbf8(gelu8(g)), u)));
  }
}
struct DCandArg { const uint16_t* emb; const int* cand; const float* dh; float* cscore; int nc; };
static void dcand_task(void* p, int tid, int T) {
  DCandArg& a = *(DCandArg*)p;
  int c0 = (int)((long)tid * a.nc / T), c1 = (int)((long)(tid + 1) * a.nc / T);
  int ci = c0;
  for (; ci + 4 <= c1; ci += 4) {
    if (ci + 8 <= c1) {
      const uint16_t* p0 = a.emb + (size_t)a.cand[ci + 4] * 256;
      const uint16_t* p1 = a.emb + (size_t)a.cand[ci + 5] * 256;
      const uint16_t* p2 = a.emb + (size_t)a.cand[ci + 6] * 256;
      const uint16_t* p3 = a.emb + (size_t)a.cand[ci + 7] * 256;
      for (int pf = 0; pf < 256; pf += 32) {
        _mm_prefetch((const char*)(p0 + pf), _MM_HINT_T0);
        _mm_prefetch((const char*)(p1 + pf), _MM_HINT_T0);
        _mm_prefetch((const char*)(p2 + pf), _MM_HINT_T0);
        _mm_prefetch((const char*)(p3 + pf), _MM_HINT_T0);
      }
    }
    const uint16_t* w0 = a.emb + (size_t)a.cand[ci + 0] * 256;
    const uint16_t* w1 = a.emb + (size_t)a.cand[ci + 1] * 256;
    const uint16_t* w2 = a.emb + (size_t)a.cand[ci + 2] * 256;
    const uint16_t* w3 = a.emb + (size_t)a.cand[ci + 3] * 256;
    __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
    for (int k2 = 0; k2 < 256; k2 += 16) {
      __m256 x0 = _mm256_loadu_ps(a.dh + k2), x1 = _mm256_loadu_ps(a.dh + k2 + 8);
      a0 = _mm256_fmadd_ps(bf8(w0 + k2), x0, a0); b0 = _mm256_fmadd_ps(bf8(w0 + k2 + 8), x1, b0);
      a1 = _mm256_fmadd_ps(bf8(w1 + k2), x0, a1); b1 = _mm256_fmadd_ps(bf8(w1 + k2 + 8), x1, b1);
      a2 = _mm256_fmadd_ps(bf8(w2 + k2), x0, a2); b2 = _mm256_fmadd_ps(bf8(w2 + k2 + 8), x1, b2);
      a3 = _mm256_fmadd_ps(bf8(w3 + k2), x0, a3); b3 = _mm256_fmadd_ps(bf8(w3 + k2 + 8), x1, b3);
    }
    a.cscore[ci + 0] = w4::hsum(_mm256_add_ps(a0, b0));
    a.cscore[ci + 1] = w4::hsum(_mm256_add_ps(a1, b1));
    a.cscore[ci + 2] = w4::hsum(_mm256_add_ps(a2, b2));
    a.cscore[ci + 3] = w4::hsum(_mm256_add_ps(a3, b3));
  }
  for (; ci < c1; ++ci) {
    const uint16_t* w0 = a.emb + (size_t)a.cand[ci] * 256;
    __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
    for (int k2 = 0; k2 < 256; k2 += 16) {
      a0 = _mm256_fmadd_ps(bf8(w0 + k2), _mm256_loadu_ps(a.dh + k2), a0);
      b0 = _mm256_fmadd_ps(bf8(w0 + k2 + 8), _mm256_loadu_ps(a.dh + k2 + 8), b0);
    }
    a.cscore[ci] = w4::hsum(_mm256_add_ps(a0, b0));
  }
}
static void b16_task(void* p, int tid, int T) {
  BArg& a = *(BArg*)p; const int total = a.cum[a.nj]; long c0 = (long)tid * total / T, c1 = (long)(tid + 1) * total / T;
  for (int j = 0; j < a.nj; ++j) { long lo = std::max<long>(c0, a.cum[j]), hi = std::min<long>(c1, a.cum[j + 1]); if (lo >= hi) continue;
    int n0 = (int)((lo - a.cum[j]) * 16), n1 = (int)std::min<long>(a.m[j]->N, (hi - a.cum[j]) * 16); b16_rows(*a.m[j], a.x, n0, n1, a.y[j]); }
}
static void b16_multi(const float* x, int nj, const B16* const* m, float* const* y) {
  BArg a; a.x = x; a.nj = nj; a.cum[0] = 0; for (int j = 0; j < nj; ++j) { a.m[j] = m[j]; a.y[j] = y[j]; a.cum[j + 1] = a.cum[j] + (m[j]->N + 15) / 16; } POOL.run(b16_task, &a);
}
static void b16_1(const float* x, const B16& m, float* y) { const B16* mm[1] = {&m}; float* yy[1] = {y}; b16_multi(x, 1, mm, yy); }
static std::map<std::string, TInfo> DIDX; static uint8_t* DBASE;
static const uint16_t* dbf(const std::string& n) { auto it = DIDX.find(n); if (it == DIDX.end()) { fprintf(stderr, "missing drafter tensor %s\n", n.c_str()); exit(1); } return (const uint16_t*)(DBASE + it->second.off); }
static B16 db16(const std::string& n) { auto it = DIDX.find(n); if (it == DIDX.end()) { fprintf(stderr, "missing drafter tensor %s\n", n.c_str()); exit(1); } return B16{(const uint16_t*)(DBASE + it->second.off), (int)it->second.shape[0], (int)it->second.shape[1]}; }
static std::vector<float> dvec(const std::string& n) { auto it = DIDX.find(n); std::vector<float> v(it->second.nbytes / 2); const uint16_t* p = (const uint16_t*)(DBASE + it->second.off); for (size_t i = 0; i < v.size(); ++i) v[i] = bf(p[i]); return v; }
struct DLayer { B16 qp, op, gate, up, down; std::vector<float> in_ln, post_attn, pre_ff, post_ff, qn; float scalar; int hd; bool sliding; };

// ---- split-KV attention for few query rows (decode / verify / drafter): every K and V row is read ONCE for all heads ----
struct AttnArg { const uint16_t* Kc; const uint16_t* Vc; int hd, nh, M, kmin, kmax; const float* q[4]; int pos[4], lo[4], skip_lo[4], skip_hi[4]; };
static inline bool attn_allowed(const AttnArg& a, int m, int p) { return p >= a.lo[m] && p <= a.pos[m] && !(p >= a.skip_lo[m] && p <= a.skip_hi[m]); }
constexpr int ACH = 8192;
alignas(64) static float A_O[16][4][8][512], A_M[16][4][8], A_L[16][4][8]; alignas(64) static float A_S[16][ACH * 4 * 8];
static void attn_task(void* p, int tid, int T) {
  AttnArg& a = *(AttnArg*)p; const int hd = a.hd, nh = a.nh, M = a.M, len = a.kmax - a.kmin + 1;
  const int c0 = a.kmin + (int)((long)tid * len / T), c1 = a.kmin + (int)((long)(tid + 1) * len / T);
  float* S = A_S[tid];
  for (int m = 0; m < M; ++m) for (int h = 0; h < nh; ++h) { A_M[tid][m][h] = -1e30f; A_L[tid][m][h] = 0; std::memset(A_O[tid][m][h], 0, hd * 4); }
  constexpr int BP = 8;
  alignas(64) float kv_tile[BP][512];
  for (int pb = c0; pb < c1; pb += BP) {
    const int nb = std::min(BP, c1 - pb);
    for (int b = 0; b < nb; ++b) {
      const uint16_t* kr = a.Kc + (size_t)(pb + b) * hd;
      float* kf = kv_tile[b];
      for (int i = 0; i < hd; i += 8) _mm256_store_ps(kf + i, bf8(kr + i));
    }
    for (int m = 0; m < M; ++m) {
      int b = 0;
      for (; b + 4 <= nb; b += 4) {
        const bool ok0 = attn_allowed(a, m, pb + b + 0);
        const bool ok1 = attn_allowed(a, m, pb + b + 1);
        const bool ok2 = attn_allowed(a, m, pb + b + 2);
        const bool ok3 = attn_allowed(a, m, pb + b + 3);
        if (ok0 && ok1 && ok2 && ok3) {
          float* sp0 = S + ((size_t)(pb + b + 0 - c0) * M + m) * 8;
          float* sp1 = S + ((size_t)(pb + b + 1 - c0) * M + m) * 8;
          float* sp2 = S + ((size_t)(pb + b + 2 - c0) * M + m) * 8;
          float* sp3 = S + ((size_t)(pb + b + 3 - c0) * M + m) * 8;
          const float* kf0 = kv_tile[b + 0];
          const float* kf1 = kv_tile[b + 1];
          const float* kf2 = kv_tile[b + 2];
          const float* kf3 = kv_tile[b + 3];
          for (int h = 0; h < nh; ++h) {
            const float* qh = a.q[m] + h * hd;
            __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps();
            __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps();
            __m256 s20 = _mm256_setzero_ps(), s21 = _mm256_setzero_ps();
            __m256 s30 = _mm256_setzero_ps(), s31 = _mm256_setzero_ps();
            for (int i = 0; i < hd; i += 16) {
              __m256 q0 = _mm256_loadu_ps(qh + i), q1 = _mm256_loadu_ps(qh + i + 8);
              s00 = _mm256_fmadd_ps(q0, _mm256_load_ps(kf0 + i), s00); s01 = _mm256_fmadd_ps(q1, _mm256_load_ps(kf0 + i + 8), s01);
              s10 = _mm256_fmadd_ps(q0, _mm256_load_ps(kf1 + i), s10); s11 = _mm256_fmadd_ps(q1, _mm256_load_ps(kf1 + i + 8), s11);
              s20 = _mm256_fmadd_ps(q0, _mm256_load_ps(kf2 + i), s20); s21 = _mm256_fmadd_ps(q1, _mm256_load_ps(kf2 + i + 8), s21);
              s30 = _mm256_fmadd_ps(q0, _mm256_load_ps(kf3 + i), s30); s31 = _mm256_fmadd_ps(q1, _mm256_load_ps(kf3 + i + 8), s31);
            }
            float sc0 = w4::hsum(_mm256_add_ps(s00, s01)); sp0[h] = sc0; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc0);
            float sc1 = w4::hsum(_mm256_add_ps(s10, s11)); sp1[h] = sc1; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc1);
            float sc2 = w4::hsum(_mm256_add_ps(s20, s21)); sp2[h] = sc2; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc2);
            float sc3 = w4::hsum(_mm256_add_ps(s30, s31)); sp3[h] = sc3; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc3);
          }
        } else {
          for (int u = 0; u < 4; ++u) {
            int pp = pb + b + u;
            if (!attn_allowed(a, m, pp)) continue;
            float* sp = S + ((size_t)(pp - c0) * M + m) * 8;
            const float* kf = kv_tile[b + u];
            for (int h = 0; h < nh; ++h) {
              const float* qh = a.q[m] + h * hd; __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
              for (int i = 0; i < hd; i += 16) { s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_load_ps(kf + i), s0); s1 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i + 8), _mm256_load_ps(kf + i + 8), s1); }
              float sc = w4::hsum(_mm256_add_ps(s0, s1)); sp[h] = sc; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc);
            }
          }
        }
      }
      for (; b < nb; ++b) {
        int pp = pb + b;
        if (!attn_allowed(a, m, pp)) continue;
        float* sp = S + ((size_t)(pp - c0) * M + m) * 8;
        const float* kf = kv_tile[b];
        for (int h = 0; h < nh; ++h) {
          const float* qh = a.q[m] + h * hd; __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
          for (int i = 0; i < hd; i += 16) { s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_load_ps(kf + i), s0); s1 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i + 8), _mm256_load_ps(kf + i + 8), s1); }
          float sc = w4::hsum(_mm256_add_ps(s0, s1)); sp[h] = sc; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc);
        }
      }
    }
  }
  __m256 mxv[4], l8[4];
  for (int m = 0; m < M; ++m) {
    float tmp8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int h = 0; h < nh; ++h) tmp8[h] = A_M[tid][m][h];
    mxv[m] = _mm256_loadu_ps(tmp8); l8[m] = _mm256_setzero_ps();
  }
  alignas(32) float wt_tile[BP][4][8];
  bool allow_tile[BP][4];
  for (int pb = c0; pb < c1; pb += BP) {
    const int nb = std::min(BP, c1 - pb);
    for (int b = 0; b < nb; ++b) {
      int pp = pb + b;
      bool any = false;
      for (int m = 0; m < M; ++m) {
        bool ok = attn_allowed(a, m, pp);
        allow_tile[b][m] = ok;
        if (ok) any = true;
      }
      if (!any) continue;
      const uint16_t* vr = a.Vc + (size_t)pp * hd;
      float* vf = kv_tile[b];
      for (int i = 0; i < hd; i += 8) _mm256_store_ps(vf + i, bf8(vr + i));
      for (int m = 0; m < M; ++m) {
        if (!allow_tile[b][m]) continue;
        __m256 w8 = exp8(_mm256_sub_ps(_mm256_loadu_ps(S + ((size_t)(pp - c0) * M + m) * 8), mxv[m]));
        l8[m] = _mm256_add_ps(l8[m], w8);
        _mm256_store_ps(wt_tile[b][m], w8);
      }
    }
    for (int m = 0; m < M; ++m) {
      for (int h = 0; h < nh; ++h) {
        float* o = A_O[tid][m][h];
        for (int i = 0; i < hd; i += 64) {
          __m256 o0 = _mm256_loadu_ps(o + i + 0),  o1 = _mm256_loadu_ps(o + i + 8);
          __m256 o2 = _mm256_loadu_ps(o + i + 16), o3 = _mm256_loadu_ps(o + i + 24);
          __m256 o4 = _mm256_loadu_ps(o + i + 32), o5 = _mm256_loadu_ps(o + i + 40);
          __m256 o6 = _mm256_loadu_ps(o + i + 48), o7 = _mm256_loadu_ps(o + i + 56);
          for (int b = 0; b < nb; ++b) {
            if (!allow_tile[b][m]) continue;
            const __m256 w = _mm256_set1_ps(wt_tile[b][m][h]);
            const float* vf = kv_tile[b] + i;
            o0 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 0),  o0);
            o1 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 8),  o1);
            o2 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 16), o2);
            o3 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 24), o3);
            o4 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 32), o4);
            o5 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 40), o5);
            o6 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 48), o6);
            o7 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 56), o7);
          }
          _mm256_storeu_ps(o + i + 0,  o0); _mm256_storeu_ps(o + i + 8,  o1);
          _mm256_storeu_ps(o + i + 16, o2); _mm256_storeu_ps(o + i + 24, o3);
          _mm256_storeu_ps(o + i + 32, o4); _mm256_storeu_ps(o + i + 40, o5);
          _mm256_storeu_ps(o + i + 48, o6); _mm256_storeu_ps(o + i + 56, o7);
        }
      }
    }
  }
  for (int m = 0; m < M; ++m) { float tl[8]; _mm256_storeu_ps(tl, l8[m]); bool any = !(a.lo[m] > a.pos[m]); for (int h = 0; h < nh; ++h) A_L[tid][m][h] = any ? tl[h] : 0.0f; }
}
#ifndef ATT_PD
#define ATT_PD 4
#endif
static void attn_task_pf(void* p, int tid, int T) {
  AttnArg& a = *(AttnArg*)p; const int hd = a.hd, nh = a.nh, M = a.M, len = a.kmax - a.kmin + 1;
  const int c0 = a.kmin + (int)((long)tid * len / T), c1 = a.kmin + (int)((long)(tid + 1) * len / T);
  float* S = A_S[tid];
  for (int m = 0; m < M; ++m) for (int h = 0; h < nh; ++h) { A_M[tid][m][h] = -1e30f; A_L[tid][m][h] = 0; std::memset(A_O[tid][m][h], 0, hd * 4); }
  constexpr int BP = 8;
  alignas(64) float kv_tile[BP][512];
  for (int pb = c0; pb < c1; pb += BP) {
    const int nb = std::min(BP, c1 - pb);
    { const int pr = pb + BP * ATT_PD; if (pr < c1) for (int b = 0; b < BP && pr + b < c1; ++b) { const char* r = (const char*)(a.Kc + (size_t)(pr + b) * hd); for (int o = 0; o < hd * 2; o += 64) _mm_prefetch(r + o, _MM_HINT_T0); } }
    for (int b = 0; b < nb; ++b) {
      const uint16_t* kr = a.Kc + (size_t)(pb + b) * hd;
      float* kf = kv_tile[b];
      for (int i = 0; i < hd; i += 8) _mm256_store_ps(kf + i, bf8(kr + i));
    }
    for (int m = 0; m < M; ++m) {
      int b = 0;
      for (; b + 4 <= nb; b += 4) {
        const bool ok0 = attn_allowed(a, m, pb + b + 0);
        const bool ok1 = attn_allowed(a, m, pb + b + 1);
        const bool ok2 = attn_allowed(a, m, pb + b + 2);
        const bool ok3 = attn_allowed(a, m, pb + b + 3);
        if (ok0 && ok1 && ok2 && ok3) {
          float* sp0 = S + ((size_t)(pb + b + 0 - c0) * M + m) * 8;
          float* sp1 = S + ((size_t)(pb + b + 1 - c0) * M + m) * 8;
          float* sp2 = S + ((size_t)(pb + b + 2 - c0) * M + m) * 8;
          float* sp3 = S + ((size_t)(pb + b + 3 - c0) * M + m) * 8;
          const float* kf0 = kv_tile[b + 0];
          const float* kf1 = kv_tile[b + 1];
          const float* kf2 = kv_tile[b + 2];
          const float* kf3 = kv_tile[b + 3];
          for (int h = 0; h < nh; ++h) {
            const float* qh = a.q[m] + h * hd;
            __m256 s00 = _mm256_setzero_ps(), s01 = _mm256_setzero_ps();
            __m256 s10 = _mm256_setzero_ps(), s11 = _mm256_setzero_ps();
            __m256 s20 = _mm256_setzero_ps(), s21 = _mm256_setzero_ps();
            __m256 s30 = _mm256_setzero_ps(), s31 = _mm256_setzero_ps();
            for (int i = 0; i < hd; i += 16) {
              __m256 q0 = _mm256_loadu_ps(qh + i), q1 = _mm256_loadu_ps(qh + i + 8);
              s00 = _mm256_fmadd_ps(q0, _mm256_load_ps(kf0 + i), s00); s01 = _mm256_fmadd_ps(q1, _mm256_load_ps(kf0 + i + 8), s01);
              s10 = _mm256_fmadd_ps(q0, _mm256_load_ps(kf1 + i), s10); s11 = _mm256_fmadd_ps(q1, _mm256_load_ps(kf1 + i + 8), s11);
              s20 = _mm256_fmadd_ps(q0, _mm256_load_ps(kf2 + i), s20); s21 = _mm256_fmadd_ps(q1, _mm256_load_ps(kf2 + i + 8), s21);
              s30 = _mm256_fmadd_ps(q0, _mm256_load_ps(kf3 + i), s30); s31 = _mm256_fmadd_ps(q1, _mm256_load_ps(kf3 + i + 8), s31);
            }
            float sc0 = w4::hsum(_mm256_add_ps(s00, s01)); sp0[h] = sc0; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc0);
            float sc1 = w4::hsum(_mm256_add_ps(s10, s11)); sp1[h] = sc1; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc1);
            float sc2 = w4::hsum(_mm256_add_ps(s20, s21)); sp2[h] = sc2; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc2);
            float sc3 = w4::hsum(_mm256_add_ps(s30, s31)); sp3[h] = sc3; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc3);
          }
        } else {
          for (int u = 0; u < 4; ++u) {
            int pp = pb + b + u;
            if (!attn_allowed(a, m, pp)) continue;
            float* sp = S + ((size_t)(pp - c0) * M + m) * 8;
            const float* kf = kv_tile[b + u];
            for (int h = 0; h < nh; ++h) {
              const float* qh = a.q[m] + h * hd; __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
              for (int i = 0; i < hd; i += 16) { s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_load_ps(kf + i), s0); s1 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i + 8), _mm256_load_ps(kf + i + 8), s1); }
              float sc = w4::hsum(_mm256_add_ps(s0, s1)); sp[h] = sc; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc);
            }
          }
        }
      }
      for (; b < nb; ++b) {
        int pp = pb + b;
        if (!attn_allowed(a, m, pp)) continue;
        float* sp = S + ((size_t)(pp - c0) * M + m) * 8;
        const float* kf = kv_tile[b];
        for (int h = 0; h < nh; ++h) {
          const float* qh = a.q[m] + h * hd; __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
          for (int i = 0; i < hd; i += 16) { s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_load_ps(kf + i), s0); s1 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i + 8), _mm256_load_ps(kf + i + 8), s1); }
          float sc = w4::hsum(_mm256_add_ps(s0, s1)); sp[h] = sc; A_M[tid][m][h] = std::max(A_M[tid][m][h], sc);
        }
      }
    }
  }
  __m256 mxv[4], l8[4];
  for (int m = 0; m < M; ++m) {
    float tmp8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int h = 0; h < nh; ++h) tmp8[h] = A_M[tid][m][h];
    mxv[m] = _mm256_loadu_ps(tmp8); l8[m] = _mm256_setzero_ps();
  }
  for (int t = 0; t < ATT_PD; ++t) { const int pr = c0 + BP * t; for (int b = 0; b < BP && pr + b < c1; ++b) { const char* r = (const char*)(a.Vc + (size_t)(pr + b) * hd); for (int o = 0; o < hd * 2; o += 64) _mm_prefetch(r + o, _MM_HINT_T0); } }
  alignas(32) float wt_tile[BP][4][8];
  bool allow_tile[BP][4];
  for (int pb = c0; pb < c1; pb += BP) {
    const int nb = std::min(BP, c1 - pb);
    { const int pr = pb + BP * ATT_PD; if (pr < c1) for (int b = 0; b < BP && pr + b < c1; ++b) { const char* r = (const char*)(a.Vc + (size_t)(pr + b) * hd); for (int o = 0; o < hd * 2; o += 64) _mm_prefetch(r + o, _MM_HINT_T0); } }
    for (int b = 0; b < nb; ++b) {
      int pp = pb + b;
      bool any = false;
      for (int m = 0; m < M; ++m) {
        bool ok = attn_allowed(a, m, pp);
        allow_tile[b][m] = ok;
        if (ok) any = true;
      }
      if (!any) continue;
      const uint16_t* vr = a.Vc + (size_t)pp * hd;
      float* vf = kv_tile[b];
      for (int i = 0; i < hd; i += 8) _mm256_store_ps(vf + i, bf8(vr + i));
      for (int m = 0; m < M; ++m) {
        if (!allow_tile[b][m]) continue;
        __m256 w8 = exp8(_mm256_sub_ps(_mm256_loadu_ps(S + ((size_t)(pp - c0) * M + m) * 8), mxv[m]));
        l8[m] = _mm256_add_ps(l8[m], w8);
        _mm256_store_ps(wt_tile[b][m], w8);
      }
    }
    for (int m = 0; m < M; ++m) {
      for (int h = 0; h < nh; ++h) {
        float* o = A_O[tid][m][h];
        for (int i = 0; i < hd; i += 64) {
          __m256 o0 = _mm256_loadu_ps(o + i + 0),  o1 = _mm256_loadu_ps(o + i + 8);
          __m256 o2 = _mm256_loadu_ps(o + i + 16), o3 = _mm256_loadu_ps(o + i + 24);
          __m256 o4 = _mm256_loadu_ps(o + i + 32), o5 = _mm256_loadu_ps(o + i + 40);
          __m256 o6 = _mm256_loadu_ps(o + i + 48), o7 = _mm256_loadu_ps(o + i + 56);
          for (int b = 0; b < nb; ++b) {
            if (!allow_tile[b][m]) continue;
            const __m256 w = _mm256_set1_ps(wt_tile[b][m][h]);
            const float* vf = kv_tile[b] + i;
            o0 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 0),  o0);
            o1 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 8),  o1);
            o2 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 16), o2);
            o3 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 24), o3);
            o4 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 32), o4);
            o5 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 40), o5);
            o6 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 48), o6);
            o7 = _mm256_fmadd_ps(w, _mm256_load_ps(vf + 56), o7);
          }
          _mm256_storeu_ps(o + i + 0,  o0); _mm256_storeu_ps(o + i + 8,  o1);
          _mm256_storeu_ps(o + i + 16, o2); _mm256_storeu_ps(o + i + 24, o3);
          _mm256_storeu_ps(o + i + 32, o4); _mm256_storeu_ps(o + i + 40, o5);
          _mm256_storeu_ps(o + i + 48, o6); _mm256_storeu_ps(o + i + 56, o7);
        }
      }
    }
  }
  for (int m = 0; m < M; ++m) { float tl[8]; _mm256_storeu_ps(tl, l8[m]); bool any = !(a.lo[m] > a.pos[m]); for (int h = 0; h < nh; ++h) A_L[tid][m][h] = any ? tl[h] : 0.0f; }
}
static void attn_merge(const AttnArg& a, int T, float* const* out, XP* xb = nullptr) {   // exact log-sum-exp merge of the per-thread partials (Lean: softmax_partial_rescale)
  const int K = a.nh * a.hd;
  if (xb) { xb->K = K; xb->M = a.M; xb->src = out[0]; xb->qdone = false; }
  auto merge_mh = [&](int mh, int) {
    const int m = mh / a.nh, h = mh % a.nh;
    float M0 = -1e30f; for (int t = 0; t < T; ++t) if (A_L[t][m][h] > 0) M0 = std::max(M0, A_M[t][m][h]);
    double l = 0; float scs[16]; int act_t[16], na = 0;
    for (int t = 0; t < T; ++t) {
      if (!(A_L[t][m][h] > 0)) continue;
      const float sc = expf(A_M[t][m][h] - M0);
      l += (double)A_L[t][m][h] * sc;
      scs[na] = sc; act_t[na] = t; ++na;
    }
    const float inv = (float)(1.0 / l); __m256 vinv = _mm256_set1_ps(inv);
    float* o = out[m] + h * a.hd;
    for (int i = 0; i < a.hd; i += 32) {
      __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
      for (int k = 0; k < na; ++k) {
        const float* src_o = &A_O[act_t[k]][m][h][i];
        __m256 vsc = _mm256_set1_ps(scs[k]);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(src_o + 0),  vsc, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(src_o + 8),  vsc, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(src_o + 16), vsc, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(src_o + 24), vsc, acc3);
      }
      acc0 = rbf8(_mm256_mul_ps(acc0, vinv)); acc1 = rbf8(_mm256_mul_ps(acc1, vinv));
      acc2 = rbf8(_mm256_mul_ps(acc2, vinv)); acc3 = rbf8(_mm256_mul_ps(acc3, vinv));
      if (xb && !G_I8_ACTIVE) {
        const int g = (h * a.hd + i) / 32;
        prep_g32_from_vecs(acc0, acc1, acc2, acc3, xb->xe + (size_t)m * (K / 2) + g * 16, xb->xo + (size_t)m * (K / 2) + g * 16, xb->gs + (size_t)m * (K / 32) + g);
      } else {
        _mm256_storeu_ps(o + i + 0,  acc0); _mm256_storeu_ps(o + i + 8,  acc1);
        _mm256_storeu_ps(o + i + 16, acc2); _mm256_storeu_ps(o + i + 24, acc3);
      }
    }
    if (xb && G_I8_ACTIVE) prep_x_avx2(o, a.hd, xb->xe + (size_t)m * (K / 2) + h * (a.hd / 2), xb->xo + (size_t)m * (K / 2) + h * (a.hd / 2), xb->gs + (size_t)m * (K / 32) + h * (a.hd / 32));
  };
  if (a.M * a.nh >= 8) pfor(a.M * a.nh, merge_mh);
  else for (int mh = 0; mh < a.M * a.nh; ++mh) merge_mh(mh, 0);
}

// ---- tiled prefill attention: QB query rows x 8 heads share every K/V row (converted bf16->f32 once per tile, not once per (row,head)) ----
// Same softmax as the per-(row,head) path; normalisation commutes with the V sum: sum_p (e_p/d) v_p = (sum_p e_p v_p)/d.
constexpr int QB = 2; static std::vector<float> TS[16];
static void attn_tile(const uint16_t* Kc, const uint16_t* Vc, int hd, bool sliding, const float* qbase, int qstride, int pos0, int M, float* attbase, int tile, int tid) {
  const int m0 = tile * QB, nm = std::min(QB, M - m0), NS = nm * 8; alignas(32) float kf[512], vf[512], wt[16];
  int lo[QB], pos[QB]; int kmin = 1 << 30, kmax = -1;
  for (int j = 0; j < nm; ++j) { pos[j] = pos0 + m0 + j; lo[j] = sliding ? std::max(0, pos[j] - 511) : 0; kmin = std::min(kmin, lo[j]); kmax = std::max(kmax, pos[j]); }
  const int len = kmax - kmin + 1; std::vector<float>& S = TS[tid]; if (S.size() < (size_t)len * 16) S.resize((size_t)len * 16);
  float mx[16], den[16]; for (int i = 0; i < 16; ++i) { mx[i] = -1e30f; den[i] = 0; }
  for (int p = kmin; p <= kmax; ++p) { const uint16_t* kr = Kc + (size_t)p * hd; for (int i = 0; i < hd; i += 8) _mm256_store_ps(kf + i, bf8(kr + i)); float* sp = S.data() + (size_t)(p - kmin) * 16;
    for (int j = 0; j < nm; ++j) { if (p < lo[j] || p > pos[j]) { for (int h = 0; h < 8; ++h) sp[j * 8 + h] = -1e30f; continue; }
      for (int h = 0; h < 8; ++h) { const float* qh = qbase + (size_t)(m0 + j) * qstride + h * hd; __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        for (int i = 0; i < hd; i += 16) { s0 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), _mm256_load_ps(kf + i), s0); s1 = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i + 8), _mm256_load_ps(kf + i + 8), s1); }
        float sc = w4::hsum(_mm256_add_ps(s0, s1)); sp[j * 8 + h] = sc; mx[j * 8 + h] = std::max(mx[j * 8 + h], sc); } } }
  alignas(32) float O[QB * 8][512]; for (int i = 0; i < NS; ++i) std::memset(O[i], 0, hd * 4);
  for (int p = kmin; p <= kmax; ++p) { bool any = false; for (int j = 0; j < nm; ++j) if (p >= lo[j] && p <= pos[j]) any = true; if (!any) continue;
    const uint16_t* vr = Vc + (size_t)p * hd; for (int i = 0; i < hd; i += 8) _mm256_store_ps(vf + i, bf8(vr + i)); const float* sp = S.data() + (size_t)(p - kmin) * 16;
    for (int j = 0; j < nm; ++j) { if (p < lo[j] || p > pos[j]) continue;
      __m256 w8 = exp8(_mm256_sub_ps(_mm256_loadu_ps(sp + j * 8), _mm256_loadu_ps(mx + j * 8))); _mm256_store_ps(wt, w8);
      for (int h = 0; h < 8; ++h) { den[j * 8 + h] += wt[h]; const __m256 w = _mm256_set1_ps(wt[h]); float* o = O[j * 8 + h]; for (int i = 0; i < hd; i += 8) _mm256_store_ps(o + i, _mm256_fmadd_ps(w, _mm256_load_ps(vf + i), _mm256_load_ps(o + i))); } } }
  for (int j = 0; j < nm; ++j) for (int h = 0; h < 8; ++h) { const float inv = 1.0f / den[j * 8 + h]; float* oh = attbase + (size_t)(m0 + j) * 8 * hd + h * hd; const float* o = O[j * 8 + h];
    for (int i = 0; i < hd; i += 8) _mm256_storeu_ps(oh + i, rbf8(_mm256_mul_ps(_mm256_load_ps(o + i), _mm256_set1_ps(inv)))); }
}
static int g_sec = -1; static long R1_n = 0, R1_top1 = 0, R1_top2 = 0, tree_attempts = 0, tree_hits = 0;
static double T_ATT = 0, T_LAYERS = 0, T_HEAD = 0, T_DRAFT = 0; static long N_FWD = 0;
// =============================== audio tower (streaming, exact; spec: proofs-cpu/CONJECTURES.md) ===============================
struct GemmArg { const uint16_t* W; const float* X; float* Y; int N, K, M; };
template <int MT> static inline void gemm_rows(const uint16_t* w, const float* X, int K, int M0, float* Y, int N, int n) {
  __m256 acc[MT]; for (int m = 0; m < MT; ++m) acc[m] = _mm256_setzero_ps();
  for (int k = 0; k < K; k += 8) { __m256 wv = bf8(w + k); for (int m = 0; m < MT; ++m) acc[m] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(X + (size_t)(M0 + m) * K + k), acc[m]); }
  for (int m = 0; m < MT; ++m) Y[(size_t)(M0 + m) * N + n] = w4::hsum(acc[m]);
}
static void gemm_task(void* p, int tid, int T) {
  GemmArg& a = *(GemmArg*)p; const int nch = (a.N + 15) / 16; long c0 = (long)tid * nch / T, c1 = (long)(tid + 1) * nch / T;
  for (long c = c0; c < c1; ++c) { int n0 = (int)(c * 16), n1 = std::min(a.N, n0 + 16);
    for (int n = n0; n < n1; ++n) { const uint16_t* w = a.W + (size_t)n * a.K; int m = 0;
      for (; m + 4 <= a.M; m += 4) gemm_rows<4>(w, a.X, a.K, m, a.Y, a.N, n);
      for (; m + 1 <= a.M; ++m) gemm_rows<1>(w, a.X, a.K, m, a.Y, a.N, n); } }
}
static void bgemm(const uint16_t* W, int N, int K, const float* X, int M, float* Y) { GemmArg a{W, X, Y, N, K, M}; POOL.run(gemm_task, &a); }
struct ALin { const uint16_t* w = nullptr; int N = 0, K = 0; float imin = -INFINITY, imax = INFINITY, omin = -INFINITY, omax = INFINITY; };
static std::vector<float> g_clamp_tmp;
static void alin(const ALin& L, const float* X, int M, float* Y) {   // clipped linear: clamp input, matmul, clamp output
  const float* xin = X; if (std::isfinite(L.imin) || std::isfinite(L.imax)) { g_clamp_tmp.resize((size_t)M * L.K); for (size_t i = 0; i < (size_t)M * L.K; ++i) g_clamp_tmp[i] = std::min(std::max(X[i], L.imin), L.imax); xin = g_clamp_tmp.data(); }
  bgemm(L.w, L.N, L.K, xin, M, Y);
  if (std::isfinite(L.omin) || std::isfinite(L.omax)) for (size_t i = 0; i < (size_t)M * L.N; ++i) Y[i] = std::min(std::max(Y[i], L.omin), L.omax);
}
static inline void rmsw(const float* x, const float* w, int n, float* y) { double ss = 0; for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i]; float r = (float)pow(ss / n + 1e-6, -0.5); for (int i = 0; i < n; ++i) y[i] = w ? x[i] * r * w[i] : x[i] * r; }
static inline float silu1(float x) { return x / (1.0f + expf(-x)); }
struct ALayer {
  ALin ff1a, ff1b, ff2a, ff2b, qp, kp, vp, post, ls, le; std::vector<float> ff1_pre, ff1_post, ff2_pre, ff2_post, n_pre, n_post, n_out, lc_pre, lc_cn, conv_w, per_dim, RK;   // RK[d][1024], d=0..11
  std::vector<float> kc, vc, gc; int nkv = 0, ng = 0;   // K/V ring (last 11 frames, in order oldest..newest), GLU history (last 4)
};
struct AudioTower {
  std::vector<ALayer> L; std::vector<float> c0w, c0n, c1w /*[co][dt][dw][ci]*/, c1n; ALin inproj, outproj, embproj; std::vector<float> outbias, mel; int emitted = 0;
  static std::vector<float> bfv(const std::string& n) { auto& t = ti(n); std::vector<float> v(t.nbytes / 2); const uint16_t* p = (const uint16_t*)(BASE + t.off); for (size_t i = 0; i < v.size(); ++i) v[i] = bf(p[i]); return v; }
  static float bfs(const std::string& n) { return bf(*(const uint16_t*)(BASE + ti(n).off)); }
  static ALin mk(const std::string& base, bool clipped) { auto& t = ti(base + (clipped ? ".linear.weight" : ".weight")); ALin l; l.w = (const uint16_t*)(BASE + t.off); l.N = (int)t.shape[0]; l.K = (int)t.shape[1];
    if (clipped) { l.imin = bfs(base + ".input_min"); l.imax = bfs(base + ".input_max"); l.omin = bfs(base + ".output_min"); l.omax = bfs(base + ".output_max"); } return l; }
  void load() {
    const std::string P = "model.audio_tower."; L.resize(12);
    // sinusoidal embeddings for distance d (0..11): [sin(d*inv) | cos(d*inv)], inv_t = exp(-t*ln(10000)/(511))
    std::vector<float> inv(512); for (int t = 0; t < 512; ++t) inv[t] = (float)exp(-(double)t * log(10000.0) / 511.0);
    for (int l = 0; l < 12; ++l) { ALayer& A = L[l]; std::string p = P + "layers." + std::to_string(l) + ".";
      A.ff1a = mk(p + "feed_forward1.ffw_layer_1", true); A.ff1b = mk(p + "feed_forward1.ffw_layer_2", true); A.ff2a = mk(p + "feed_forward2.ffw_layer_1", true); A.ff2b = mk(p + "feed_forward2.ffw_layer_2", true);
      A.qp = mk(p + "self_attn.q_proj", true); A.kp = mk(p + "self_attn.k_proj", true); A.vp = mk(p + "self_attn.v_proj", true); A.post = mk(p + "self_attn.post", true); A.ls = mk(p + "lconv1d.linear_start", true); A.le = mk(p + "lconv1d.linear_end", true);
      A.ff1_pre = bfv(p + "feed_forward1.pre_layer_norm.weight"); A.ff1_post = bfv(p + "feed_forward1.post_layer_norm.weight"); A.ff2_pre = bfv(p + "feed_forward2.pre_layer_norm.weight"); A.ff2_post = bfv(p + "feed_forward2.post_layer_norm.weight");
      A.n_pre = bfv(p + "norm_pre_attn.weight"); A.n_post = bfv(p + "norm_post_attn.weight"); A.n_out = bfv(p + "norm_out.weight"); A.lc_pre = bfv(p + "lconv1d.pre_layer_norm.weight"); A.lc_cn = bfv(p + "lconv1d.conv_norm.weight"); A.conv_w = bfv(p + "lconv1d.depthwise_conv1d.weight");
      auto pd = bfv(p + "self_attn.per_dim_scale"); A.per_dim.resize(128); for (int d = 0; d < 128; ++d) A.per_dim[d] = (float)log1p(exp((double)pd[d]));   // softplus
      const uint16_t* rkw = (const uint16_t*)(BASE + ti(p + "self_attn.relative_k_proj.weight").off); A.RK.assign(12 * 1024, 0);
      for (int d = 0; d < 12; ++d) { std::vector<float> pe(1024); for (int t = 0; t < 512; ++t) { pe[t] = (float)sin((double)d * inv[t]); pe[512 + t] = (float)cos((double)d * inv[t]); }
        for (int n = 0; n < 1024; ++n) { double s = 0; for (int k = 0; k < 1024; ++k) s += (double)bf(rkw[(size_t)n * 1024 + k]) * pe[k]; A.RK[d * 1024 + n] = (float)s; } }
      A.kc.assign(11 * 1024, 0); A.vc.assign(11 * 1024, 0); A.gc.assign(4 * 1024, 0); }
    c0w = bfv(P + "subsample_conv_projection.layer0.conv.weight"); c0n = bfv(P + "subsample_conv_projection.layer0.norm.weight"); c1n = bfv(P + "subsample_conv_projection.layer1.norm.weight");
    { auto w = bfv(P + "subsample_conv_projection.layer1.conv.weight"); c1w.assign(32 * 9 * 128, 0); for (int co = 0; co < 32; ++co) for (int ci = 0; ci < 128; ++ci) for (int dt = 0; dt < 3; ++dt) for (int dw = 0; dw < 3; ++dw) c1w[((co * 3 + dt) * 3 + dw) * 128 + ci] = w[((co * 128 + ci) * 3 + dt) * 3 + dw]; }
    inproj = mk(P + "subsample_conv_projection.input_proj_linear", false); outproj = mk(P + "output_proj", false); outbias = bfv(P + "output_proj.bias"); embproj = mk("model.embed_audio.embedding_projection", false);
  }
  void reset() { mel.clear(); emitted = 0; for (auto& A : L) { A.nkv = 0; A.ng = 0; std::fill(A.gc.begin(), A.gc.end(), 0.0f); } }
  // audio frames [t0,t1) from the mel window (zero padding only at the window edges, which lie outside the needed outputs unless at a true stream edge)
  void subsample(int t0, int t1, int nmel, float* out) {
    int m0 = std::max(0, 4 * t0 - 3 - 16); m0 -= m0 % 4; int m1 = std::min(nmel, 4 * t1 + 3 + 16); const int Tm = m1 - m0; const float* X = mel.data() + (size_t)m0 * 128;
    const int T1 = (Tm + 1) / 2, T2 = (T1 + 1) / 2; std::vector<float> o0((size_t)T1 * 64 * 128);
    for (int u = 0; u < T1; ++u) for (int w = 0; w < 64; ++w) { float* o = &o0[((size_t)u * 64 + w) * 128];
      for (int co = 0; co < 128; ++co) { float s = 0; for (int dt = 0; dt < 3; ++dt) { int r = 2 * u - 1 + dt; if (r < 0 || r >= Tm) continue; for (int dw = 0; dw < 3; ++dw) { int c = 2 * w - 1 + dw; if (c < 0 || c >= 128) continue; s += X[(size_t)r * 128 + c] * c0w[(co * 3 + dt) * 3 + dw]; } } o[co] = s; }
      double mu = 0; for (int co = 0; co < 128; ++co) mu += o[co]; mu /= 128; double var = 0; for (int co = 0; co < 128; ++co) var += (o[co] - mu) * (o[co] - mu); var /= 128; float rs = (float)(1.0 / sqrt(var + 1e-6));
      for (int co = 0; co < 128; ++co) { float v = (float)((o[co] - mu) * rs) * c0n[co]; o[co] = v > 0 ? v : 0; } }
    std::vector<float> feat((size_t)T2 * 1024);
    for (int t = 0; t < T2; ++t) for (int f = 0; f < 32; ++f) { float acc[32]; for (int co = 0; co < 32; ++co) acc[co] = 0;
      for (int dt = 0; dt < 3; ++dt) { int r = 2 * t - 1 + dt; if (r < 0 || r >= T1) continue; for (int dw = 0; dw < 3; ++dw) { int c = 2 * f - 1 + dw; if (c < 0 || c >= 64) continue; const float* in = &o0[((size_t)r * 64 + c) * 128];
          for (int co = 0; co < 32; ++co) { const float* wv = &c1w[((co * 3 + dt) * 3 + dw) * 128]; __m256 s = _mm256_setzero_ps(); for (int ci = 0; ci < 128; ci += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(in + ci), _mm256_loadu_ps(wv + ci), s); acc[co] += w4::hsum(s); } } }
      double mu = 0; for (int co = 0; co < 32; ++co) mu += acc[co]; mu /= 32; double var = 0; for (int co = 0; co < 32; ++co) var += (acc[co] - mu) * (acc[co] - mu); var /= 32; float rs = (float)(1.0 / sqrt(var + 1e-6));
      for (int co = 0; co < 32; ++co) { float v = (float)((acc[co] - mu) * rs) * c1n[co]; feat[(size_t)t * 1024 + f * 32 + co] = v > 0 ? v : 0; } }
    const int first = m0 / 4; std::vector<float> sel((size_t)(t1 - t0) * 1024); for (int i = 0; i < t1 - t0; ++i) std::memcpy(&sel[(size_t)i * 1024], &feat[(size_t)(t0 - first + i) * 1024], 4096);
    bgemm(inproj.w, 1024, 1024, sel.data(), t1 - t0, out);
  }
  void ff(const ALin& a, const ALin& b, const std::vector<float>& pre, const std::vector<float>& post, float* x, int M) {   // macaron feed-forward with residual 0.5
    std::vector<float> n((size_t)M * 1024), h((size_t)M * 4096), o((size_t)M * 1024);
    for (int m = 0; m < M; ++m) rmsw(x + (size_t)m * 1024, pre.data(), 1024, &n[(size_t)m * 1024]);
    alin(a, n.data(), M, h.data()); for (size_t i = 0; i < h.size(); ++i) h[i] = silu1(h[i]); alin(b, h.data(), M, o.data());
    for (int m = 0; m < M; ++m) { float* om = &o[(size_t)m * 1024]; std::vector<float> t(1024); rmsw(om, post.data(), 1024, t.data()); for (int i = 0; i < 1024; ++i) x[(size_t)m * 1024 + i] += 0.5f * t[i]; }
  }
  void layer(ALayer& A, float* x, int M) {   // x: [M,1024] in/out
    ff(A.ff1a, A.ff1b, A.ff1_pre, A.ff1_post, x, M);
    std::vector<float> res(x, x + (size_t)M * 1024), n((size_t)M * 1024), q((size_t)M * 1024), k((size_t)M * 1024), v((size_t)M * 1024), ctx((size_t)M * 1024), ao((size_t)M * 1024);
    for (int m = 0; m < M; ++m) rmsw(x + (size_t)m * 1024, A.n_pre.data(), 1024, &n[(size_t)m * 1024]);
    alin(A.qp, n.data(), M, q.data()); alin(A.kp, n.data(), M, k.data()); alin(A.vp, n.data(), M, v.data());
    const float qscale = (float)(pow(128.0, -0.5) / log(2.0)), kscale = (float)(log(1.0 + exp(1.0)) / log(2.0));
    for (int m = 0; m < M; ++m) for (int h = 0; h < 8; ++h) for (int d = 0; d < 128; ++d) { q[(size_t)m * 1024 + h * 128 + d] *= qscale * A.per_dim[d]; k[(size_t)m * 1024 + h * 128 + d] *= kscale; }
    // key/value sequence = cached frames then the new ones
    const int NC = A.nkv; std::vector<float> K((size_t)(NC + M) * 1024), V((size_t)(NC + M) * 1024);
    std::memcpy(K.data(), A.kc.data() + (size_t)(11 - NC) * 1024, (size_t)NC * 4096); std::memcpy(V.data(), A.vc.data() + (size_t)(11 - NC) * 1024, (size_t)NC * 4096);
    std::memcpy(K.data() + (size_t)NC * 1024, k.data(), (size_t)M * 4096); std::memcpy(V.data() + (size_t)NC * 1024, v.data(), (size_t)M * 4096);
    for (int m = 0; m < M; ++m) { const int i = NC + m, j0 = std::max(0, i - 11);
      for (int h = 0; h < 8; ++h) { const float* qh = &q[(size_t)m * 1024 + h * 128]; float sc[12], mx = -1e30f;
        for (int j = j0; j <= i; ++j) { const float* kh = &K[(size_t)j * 1024 + h * 128]; const float* rk = &A.RK[(size_t)(i - j) * 1024 + h * 128]; float s = 0; for (int d = 0; d < 128; ++d) s += qh[d] * (kh[d] + rk[d]);
          s = tanhf(s / 50.0f) * 50.0f; sc[j - j0] = s; mx = std::max(mx, s); }
        float den = 0; for (int j = j0; j <= i; ++j) { sc[j - j0] = expf(sc[j - j0] - mx); den += sc[j - j0]; }
        float* c = &ctx[(size_t)m * 1024 + h * 128]; for (int d = 0; d < 128; ++d) c[d] = 0; for (int j = j0; j <= i; ++j) { const float w = sc[j - j0] / den; const float* vh = &V[(size_t)j * 1024 + h * 128]; for (int d = 0; d < 128; ++d) c[d] += w * vh[d]; } } }
    { int tot = NC + M, keep = std::min(11, tot); std::memcpy(A.kc.data() + (size_t)(11 - keep) * 1024, K.data() + (size_t)(tot - keep) * 1024, (size_t)keep * 4096); std::memcpy(A.vc.data() + (size_t)(11 - keep) * 1024, V.data() + (size_t)(tot - keep) * 1024, (size_t)keep * 4096); A.nkv = keep; }
    alin(A.post, ctx.data(), M, ao.data());
    for (int m = 0; m < M; ++m) { std::vector<float> t(1024); rmsw(&ao[(size_t)m * 1024], A.n_post.data(), 1024, t.data()); for (int i = 0; i < 1024; ++i) x[(size_t)m * 1024 + i] = res[(size_t)m * 1024 + i] + t[i]; }
    // light conv
    { std::vector<float> r2(x, x + (size_t)M * 1024), nn((size_t)M * 1024), g2((size_t)M * 2048), glu((size_t)M * 1024), cv((size_t)M * 1024), le((size_t)M * 1024);
      for (int m = 0; m < M; ++m) rmsw(x + (size_t)m * 1024, A.lc_pre.data(), 1024, &nn[(size_t)m * 1024]);
      alin(A.ls, nn.data(), M, g2.data()); for (int m = 0; m < M; ++m) for (int c = 0; c < 1024; ++c) { float a = g2[(size_t)m * 2048 + c], b = g2[(size_t)m * 2048 + 1024 + c]; glu[(size_t)m * 1024 + c] = a / (1.0f + expf(-b)); }
      const int NG = A.ng; std::vector<float> G((size_t)(NG + M) * 1024); std::memcpy(G.data(), A.gc.data() + (size_t)(4 - NG) * 1024, (size_t)NG * 4096); std::memcpy(G.data() + (size_t)NG * 1024, glu.data(), (size_t)M * 4096);
      for (int m = 0; m < M; ++m) for (int c = 0; c < 1024; ++c) { float s = 0; for (int t = 0; t < 5; ++t) { int idx = NG + m - 4 + t; if (idx >= 0) s += conv_at(A, c, t) * G[(size_t)idx * 1024 + c]; } cv[(size_t)m * 1024 + c] = s; }
      { int tot = NG + M, keep = std::min(4, tot); std::memcpy(A.gc.data() + (size_t)(4 - keep) * 1024, G.data() + (size_t)(tot - keep) * 1024, (size_t)keep * 4096); A.ng = keep; }
      for (int m = 0; m < M; ++m) { std::vector<float> t(1024); rmsw(&cv[(size_t)m * 1024], A.lc_cn.data(), 1024, t.data()); for (int c = 0; c < 1024; ++c) cv[(size_t)m * 1024 + c] = silu1(t[c]); }
      alin(A.le, cv.data(), M, le.data()); for (size_t i = 0; i < (size_t)M * 1024; ++i) x[i] = le[i] + r2[i]; }
    ff(A.ff2a, A.ff2b, A.ff2_pre, A.ff2_post, x, M);
    for (int m = 0; m < M; ++m) { std::vector<float> t(1024); rmsw(x + (size_t)m * 1024, A.n_out.data(), 1024, t.data()); std::memcpy(x + (size_t)m * 1024, t.data(), 4096); }
  }
  static float conv_at(ALayer& A, int c, int t) { return A.conv_w[(size_t)c * 5 + t]; }
  // feed new mel frames; returns the new final embedding rows [n,1536]
  std::vector<float> feed(const float* newmel, int nf, bool final) {
    mel.insert(mel.end(), newmel, newmel + (size_t)nf * 128); const int nmel = (int)(mel.size() / 128);
    int avail = final ? (nmel + 3) / 4 : (nmel >= 4 ? (nmel - 4) / 4 + 1 : 0); if (avail <= emitted) return {};
    const int M = avail - emitted; std::vector<float> x((size_t)M * 1024); subsample(emitted, avail, nmel, x.data());
    for (auto& A : L) layer(A, x.data(), M);
    std::vector<float> o((size_t)M * 1536), e((size_t)M * 1536), out((size_t)M * 1536); bgemm(outproj.w, 1536, 1024, x.data(), M, o.data());
    for (int m = 0; m < M; ++m) for (int i = 0; i < 1536; ++i) o[(size_t)m * 1536 + i] += outbias[i];
    for (int m = 0; m < M; ++m) rmsw(&o[(size_t)m * 1536], nullptr, 1536, &e[(size_t)m * 1536]);
    bgemm(embproj.w, 1536, 1536, e.data(), M, out.data()); emitted = avail; return out;
  }
};
// ============================================================================================================================

int main(int argc, char** argv) {
  G_I8 = getenv("ENGINE_PREFILL_INT8") && atoi(getenv("ENGINE_PREFILL_INT8")) == 1;
  std::string ck = argv[1], idxf = argv[2], promptf = argv[3]; int nnew = atoi(argv[4]); const int ctxmax = getenv("ENGINE_CTX") ? atoi(getenv("ENGINE_CTX")) : 4096; const int FAKE_N = getenv("ENGINE_FAKE_CTX") ? atoi(getenv("ENGINE_FAKE_CTX")) : 0;
  const bool SPLITKV = !(getenv("ENGINE_SPLITKV") && atoi(getenv("ENGINE_SPLITKV")) == 0);
  const bool TILED = !getenv("ENGINE_NO_TILED");
  const int ATT_PAR = getenv("ENGINE_ATT_PAR") ? atoi(getenv("ENGINE_ATT_PAR")) : 192;
  const int PFB = getenv("ENGINE_PREFILL_BLOCK") ? atoi(getenv("ENGINE_PREFILL_BLOCK")) : 32;
  std::vector<int> cpus = {0, 2, 4, 6, 8, 10}; if (const char* c = getenv("ENGINE_CPUS")) { cpus.clear(); std::stringstream ss(c); std::string t; while (std::getline(ss, t, ',')) cpus.push_back(atoi(t.c_str())); }
  POOL.start(cpus);
  { std::ifstream f(idxf); std::string line; while (std::getline(f, line)) { std::stringstream ss(line); std::string n, d, o, b, sh; std::getline(ss, n, '\t'); std::getline(ss, d, '\t'); std::getline(ss, o, '\t'); std::getline(ss, b, '\t'); std::getline(ss, sh, '\t');
      TInfo t; t.dtype = d; t.off = std::stoull(o); t.nbytes = std::stoull(b); std::stringstream s2(sh); std::string x; while (std::getline(s2, x, ',')) t.shape.push_back(std::stoll(x)); IDX[n] = t; } }
  int fd = open((ck + "/model.safetensors").c_str(), O_RDONLY); struct stat st; fstat(fd, &st);
  BASE = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
  const int H = 1536, NL = 35, PD = 256; std::vector<int> prompt; { std::ifstream f(promptf); int v; while (f >> v) prompt.push_back(v); }
  std::vector<Layer> L(NL);
  for (int l = 0; l < NL; ++l) { Layer& a = L[l]; std::string p = "layers." + std::to_string(l) + ".";
    a.sliding = (l % 5 != 4); a.hd = a.sliding ? 256 : 512; a.owner = l < 15;
    a.qp = q4(p + "self_attn.q_proj"); if (a.owner) { a.kp = q4(p + "self_attn.k_proj"); a.vp = q4(p + "self_attn.v_proj"); a.kn = bfvec(p + "self_attn.k_norm.weight"); }
    a.op = q4(p + "self_attn.o_proj"); a.gate = q4(p + "mlp.gate_proj"); a.up = q4(p + "mlp.up_proj"); a.down = q4(p + "mlp.down_proj"); a.pg = q4(p + "per_layer_input_gate"); a.pp = q4(p + "per_layer_projection");
    a.in_ln = bfvec(p + "input_layernorm.weight"); a.post_attn = bfvec(p + "post_attention_layernorm.weight"); a.pre_ff = bfvec(p + "pre_feedforward_layernorm.weight");
    a.post_ff = bfvec(p + "post_feedforward_layernorm.weight"); a.post_ple = bfvec(p + "post_per_layer_input_norm.weight"); a.qn = bfvec(p + "self_attn.q_norm.weight"); a.scalar = bf(bfp(p + "layer_scalar")[0]); }
  Q4 plproj = q4("per_layer_model_projection"); std::vector<float> plnorm = bfvec("per_layer_projection_norm.weight"), fnorm = bfvec("norm.weight");
  const uint16_t* E = bfp("embed_tokens.weight"); const uint16_t* PLE = bfp("embed_tokens_per_layer.weight");
  std::vector<int64_t> hist(256, 0); for (size_t i = 0; i < (size_t)262144 * H; ++i) hist[(E[i] >> 7) & 0xFF]++;
  std::vector<int> ord(256); for (int i = 0; i < 256; ++i) ord[i] = i; std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return hist[a] > hist[b]; });
  uint8_t cb8[8]; for (int i = 0; i < 8; ++i) cb8[i] = (uint8_t)ord[i]; bf16p::Packed HP; bf16p::pack(E, 262144, H, cb8, HP); mlock(HP.data.data(), HP.data.size());
  SCR.on = getenv("ENGINE_HEAD_SCREEN") && atoi(getenv("ENGINE_HEAD_SCREEN")) == 1;
  if (SCR.on) { auto tb0 = std::chrono::steady_clock::now(); SCR.w.assign((size_t)262144 * 768, 0); SCR.s.assign((size_t)262144 * 48, 0); SCR.coef.assign(262144, 0);
    struct BA { Screen* S; const uint16_t* E; } ba{&SCR, E};
    POOL.run([](void* pp, int tid, int T) { BA& b = *(BA*)pp; long r0 = (long)tid * 262144 / T, r1 = (long)(tid + 1) * 262144 / T;
      for (long n = r0; n < r1; ++n) { double e2 = 0, w2 = 0; for (int g = 0; g < 48; ++g) { float amax = 0, w[32]; for (int i = 0; i < 32; ++i) { w[i] = bf(b.E[(size_t)n * 1536 + g * 32 + i]); amax = std::max(amax, std::fabs(w[i])); }
          uint16_t sb = f2bf(amax / 7.0f); float sc = bf(sb); b.S->s[(size_t)n * 48 + g] = sb; int q[32];
          for (int i = 0; i < 32; ++i) { int v = sc > 0 ? (int)std::nearbyintf(w[i] / sc) : 0; v = std::max(-8, std::min(7, v)); q[i] = v + 8; double d = (double)w[i] - (double)v * sc; e2 += d * d; w2 += (double)w[i] * w[i]; }
          for (int i = 0; i < 16; ++i) b.S->w[(size_t)n * 768 + g * 16 + i] = (uint8_t)(q[2 * i] | (q[2 * i + 1] << 4)); }
        b.S->coef[n] = (float)(sqrt(e2) + 1e-4 * sqrt(w2)); } }, &ba);
    fprintf(stderr, "head screen built: %.1f MB int4 + %.1f MB scales, %.2fs\n", SCR.w.size() / 1e6, SCR.s.size() * 2 / 1e6, std::chrono::duration<double>(std::chrono::steady_clock::now() - tb0).count()); }
  static long scr_rows = 0, scr_surv = 0, scr_max = 0, scr_fallback = 0;
  std::vector<float> RC[2], RS[2]; for (int ty = 0; ty < 2; ++ty) { int hd = ty ? 512 : 256; double theta = ty ? 1e6 : 1e4; int ra = (int)(hd * (ty ? 0.25 : 1.0)) / 2; RC[ty].resize((size_t)ctxmax * hd / 2); RS[ty].resize((size_t)ctxmax * hd / 2);
    for (int p = 0; p < ctxmax; ++p) for (int j = 0; j < hd / 2; ++j) { double inv = j < ra ? 1.0 / pow(theta, (2.0 * j) / hd) : 0.0; double ang = p * inv; RC[ty][(size_t)p * hd / 2 + j] = (float)cos(ang); RS[ty][(size_t)p * hd / 2 + j] = (float)sin(ang); } }
  std::vector<std::vector<uint16_t>> KC(15), VC(15); for (int l = 0; l < 15; ++l) { KC[l].assign((size_t)ctxmax * L[l].hd, 0); VC[l].assign((size_t)ctxmax * L[l].hd, 0); }
  if (FAKE_N > 0) { uint32_t rs = 12345; auto rnd = [&]() { rs = rs * 1664525u + 1013904223u; return ((int)(rs >> 9) % 2001 - 1000) / 700.0f; };
    for (int l = 0; l < 15; ++l) for (size_t i = 0; i < (size_t)FAKE_N * L[l].hd; ++i) { KC[l][i] = f2bf(rnd()); VC[l][i] = f2bf(rnd()); } }
  auto owner_of = [&](int l) { if (l < 15) return l; return (l % 5 != 4) ? 13 : 14; };
  alignas(64) static float x[MM][1536], a[MM][1536], tmp[1536], xr[MM][1536], mlp_in[MM][1536], o[MM][1536], hbuf[MM][1536], g2[MM][256], pgv[MM][256], pj[35 * 256], pli[MM][35 * 256], pr[MM][35 * 256], q[MM][8 * 512], k[MM][512], v[MM][512], att[MM][8 * 512], gate[MM][12288], up[MM][12288], act[MM][12288], sc[4096], tbuf[512];
  alignas(64) static float logit[4][262144]; static XP XA, XB; static HeadArg HA; static int G[MM];

  // forward M tokens at positions pos0..pos0+M-1; if want, fills G[m] (greedy next token after row m) and hbuf[m] (final-normed hidden)
  auto fwd = [&](const int* toks, int M, int pos0, bool want, int hf = 0, const float* const* emb = nullptr, int sibling = -1) {
    if (getenv("ENGINE_ADAPTIVE_THREADS")) { const int cut = getenv("ENGINE_THREAD_CUTOFF") ? atoi(getenv("ENGINE_THREAD_CUTOFF")) : ATT_PAR; POOL.set_active(pos0 + M > cut ? 8 : 6); }
    auto PF = [&](int n, auto&& f) { if (n >= 2) pfor(n, f); else for (int i = 0; i < n; ++i) f(i, 0); };
    const __m256 vsqh = _mm256_set1_ps(sqrtf((float)H));
    XB.K = H; XB.M = M; XB.src = &x[0][0]; XB.qdone = false;
    XA.K = H; XA.M = M; XA.src = &a[0][0]; XA.qdone = false;
    PF(M, [&](int m, int) {
      if (emb && emb[m]) { for (int i = 0; i < H; i += 8) _mm256_storeu_ps(x[m] + i, rbf8(_mm256_loadu_ps(emb[m] + i))); }
      else { const uint16_t* er = E + (size_t)toks[m] * H; for (int i = 0; i < H; i += 8) _mm256_storeu_ps(x[m] + i, rbf8(_mm256_mul_ps(bf8(er + i), vsqh))); }
      prep_x_avx2(x[m], H, XB.xe + (size_t)m * (H / 2), XB.xo + (size_t)m * (H / 2), XB.gs + (size_t)m * (H / 32));
      rms_prep_x(x[m], L[0].in_ln.data(), H, G_I8_ACTIVE ? a[m] : nullptr, XA.xe + (size_t)m * (H / 2), XA.xo + (size_t)m * (H / 2), XA.gs + (size_t)m * (H / 32), false);
    });
    if (!G_I8_ACTIVE) {
      PleFusedArg pfa{&XB, &plproj, plnorm.data(), PLE, toks, emb, &pr[0][0], &pli[0][0], M, NL, PD, H};
      POOL.run(ple_fused_task, &pfa);
    } else {
      lin1(XB, plproj, &pr[0][0]); const float s1 = 1.0f / sqrtf((float)H); const __m256 vs1 = _mm256_set1_ps(s1), v16 = _mm256_set1_ps(16.0f), vinv2 = _mm256_set1_ps(0.70710678118f);
      pfor(M * NL, [&](int ml, int) {
        const int m = ml / NL, l = ml % NL;
        float* prl = pr[m] + l * PD;
        for (int i = 0; i < PD; i += 8) _mm256_storeu_ps(prl + i, rbf8(_mm256_mul_ps(_mm256_loadu_ps(prl + i), vs1)));
        alignas(32) float pjl[256];
        rms(prl, plnorm.data(), PD, pjl);
        const uint16_t* pel = PLE + (size_t)((emb && emb[m]) ? 0 : toks[m]) * 8960 + l * PD;
        float* plil = pli[m] + l * PD;
        for (int i = 0; i < PD; i += 8) {
          __m256 pev = rbf8(_mm256_mul_ps(bf8(pel + i), v16));
          _mm256_storeu_ps(plil + i, _mm256_mul_ps(_mm256_add_ps(_mm256_load_ps(pjl + i), pev), vinv2));
        }
      });
    }
    auto tl0 = std::chrono::steady_clock::now(); ++N_FWD;
    for (int l = 0; l < NL; ++l) { Layer& A = L[l]; const int hd = A.hd, ty = A.sliding ? 0 : 1;
      if (A.owner) { const Q4* qq[3] = {&A.qp, &A.kp, &A.vp}; float* yy[3] = {&q[0][0], &k[0][0], &v[0][0]}; lin_multi(XA, 3, qq, yy); } else lin1(XA, A.qp, &q[0][0]);
      if (A.owner && !(M <= 4 && pos0 + M > ATT_PAR && SPLITKV)) {
        PF(M, [&](int m, int) {
          const int cachepos = pos0 + m, pos = m == sibling ? pos0 + 1 : cachepos;
          const float* cs = &RC[ty][(size_t)pos * hd / 2]; const float* sn = &RS[ty][(size_t)pos * hd / 2];
          float* km = &k[0][0] + (size_t)m * A.kp.N; float* vm = &v[0][0] + (size_t)m * A.vp.N;
          rms_r16_inplace(km, A.kn.data(), hd); rope_avx2(km, cs, sn, hd);
          rms_r16_inplace(vm, nullptr, hd);
          uint16_t* kdst = &KC[l][(size_t)cachepos * hd]; uint16_t* vdst = &VC[l][(size_t)cachepos * hd];
          for (int i = 0; i < hd; i += 8) {
            _mm_storeu_si128((__m128i*)(kdst + i), f2bf8(_mm256_loadu_ps(km + i)));
            _mm_storeu_si128((__m128i*)(vdst + i), f2bf8(_mm256_loadu_ps(vm + i)));
          }
        });
      }
      const int ol = owner_of(l); const uint16_t* Kc = KC[ol].data(); const uint16_t* Vc = VC[ol].data();
      if (M <= 4 && pos0 + M > ATT_PAR && SPLITKV) {
        const int h_per_m = A.owner ? 9 : 8;
        pfor(M * h_per_m, [&](int idx, int) {
          const int m = idx / h_per_m, h = idx % h_per_m;
          const int cachepos = pos0 + m, pos = m == sibling ? pos0 + 1 : cachepos;
          const float* cs = &RC[ty][(size_t)pos * hd / 2]; const float* sn = &RS[ty][(size_t)pos * hd / 2];
          if (h < 8) {
            float* qh = &q[0][0] + (size_t)m * A.qp.N + h * hd;
            rms_r16_inplace(qh, A.qn.data(), hd); rope_avx2(qh, cs, sn, hd);
          } else {
            float* km = &k[0][0] + (size_t)m * A.kp.N; float* vm = &v[0][0] + (size_t)m * A.vp.N;
            rms_r16_inplace(km, A.kn.data(), hd); rope_avx2(km, cs, sn, hd);
            rms_r16_inplace(vm, nullptr, hd);
            uint16_t* kdst = &KC[l][(size_t)cachepos * hd]; uint16_t* vdst = &VC[l][(size_t)cachepos * hd];
            for (int i = 0; i < hd; i += 8) {
              _mm_storeu_si128((__m128i*)(kdst + i), f2bf8(_mm256_loadu_ps(km + i)));
              _mm_storeu_si128((__m128i*)(vdst + i), f2bf8(_mm256_loadu_ps(vm + i)));
            }
          }
        });
        AttnArg aa; aa.Kc = Kc; aa.Vc = Vc; aa.hd = hd; aa.nh = 8; aa.M = M; aa.kmax = pos0 + M - 1; aa.kmin = aa.kmax; float* outp[4];
        for (int m = 0; m < M; ++m) { aa.q[m] = &q[0][0] + (size_t)m * A.qp.N; aa.pos[m] = pos0 + m; aa.lo[m] = A.sliding ? std::max(0, (m == sibling ? pos0 + 1 : pos0 + m) - 511) : 0; aa.skip_lo[m] = m == sibling ? pos0 + 1 : 1; aa.skip_hi[m] = m == sibling ? pos0 + m - 1 : 0; aa.kmin = std::min(aa.kmin, aa.lo[m]); outp[m] = &att[0][0] + (size_t)m * 8 * hd; }
        auto ta0 = std::chrono::steady_clock::now(); POOL.run(attn_task, &aa); auto ta1 = std::chrono::steady_clock::now(); attn_merge(aa, POOL.active_count(), outp, &XB); T_ATT += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta0).count(); (void)ta1;
      } else if (sibling < 0 && TILED && M >= 4 && pos0 + M > 64) {
        pfor(M, [&](int m, int) {
          const int cachepos = pos0 + m, pos = cachepos;
          const float* cs = &RC[ty][(size_t)pos * hd / 2]; const float* sn = &RS[ty][(size_t)pos * hd / 2];
          float* qm = &q[0][0] + (size_t)m * A.qp.N;
          for (int h = 0; h < 8; ++h) { float* qh = qm + h * hd; rms_r16_inplace(qh, A.qn.data(), hd); rope_avx2(qh, cs, sn, hd); }
        });
        pfor((M + QB - 1) / QB, [&](int t, int tid) { attn_tile(Kc, Vc, hd, A.sliding, &q[0][0], A.qp.N, pos0, M, &att[0][0], t, tid); });
        XB.prep(&att[0][0], 8 * hd, M);
      } else {
        XB.K = 8 * hd; XB.M = M; XB.src = &att[0][0]; XB.qdone = false;
        auto run_mh = [&](int mh, int tid) {
          const int m = mh / 8, h = mh % 8, cachepos = pos0 + m, pos = m == sibling ? pos0 + 1 : cachepos;
          const float* cs = &RC[ty][(size_t)pos * hd / 2]; const float* sn = &RS[ty][(size_t)pos * hd / 2];
          float* qh = &q[0][0] + (size_t)m * A.qp.N + h * hd;
          rms_r16_inplace(qh, A.qn.data(), hd); rope_avx2(qh, cs, sn, hd);
          float* sc = SC[tid]; int lo = A.sliding ? std::max(0, pos - 511) : 0;
          auto valid = [&](int p) { return m != sibling || p <= pos0 || p == cachepos; }; float mx = -1e30f;
          for (int p = lo; p <= cachepos; ++p) { if (!valid(p)) continue; __m256 sv = _mm256_setzero_ps(); const uint16_t* kp = Kc + (size_t)p * hd; for (int i = 0; i < hd; i += 8) sv = _mm256_fmadd_ps(_mm256_loadu_ps(qh + i), bf8(kp + i), sv); float d = w4::hsum(sv); sc[p] = d; mx = std::max(mx, d); }
          float den = 0; for (int p = lo; p <= cachepos; ++p) { if (!valid(p)) continue; sc[p] = expf(sc[p] - mx); den += sc[p]; }
          float* oh = &att[0][0] + (size_t)m * 8 * hd + h * hd; for (int i = 0; i < hd; ++i) oh[i] = 0;
          for (int p = lo; p <= cachepos; ++p) { if (!valid(p)) continue; float wgt = sc[p] / den; const uint16_t* vp = Vc + (size_t)p * hd; for (int i = 0; i < hd; i += 8) _mm256_storeu_ps(oh + i, _mm256_fmadd_ps(_mm256_set1_ps(wgt), bf8(vp + i), _mm256_loadu_ps(oh + i))); }
          rbf_n(oh, hd);
          prep_x_avx2(oh, hd, XB.xe + (size_t)m * (4 * hd) + h * (hd / 2), XB.xo + (size_t)m * (4 * hd) + h * (hd / 2), XB.gs + (size_t)m * (8 * hd / 32) + h * (hd / 32));
        };
        if (M >= 2 || pos0 + M > ATT_PAR) pfor(M * 8, run_mh);
        else for (int mh = 0; mh < M * 8; ++mh) run_mh(mh, 0);
      }
      lin1(XB, A.op, &o[0][0]);
      XA.K = H; XA.M = M; XA.src = &mlp_in[0][0]; XA.qdone = false;
      PF(M, [&](int m, int tid) {
        float* tmp = TMP[tid]; rbf_n(o[m], H); rms(o[m], A.post_attn.data(), H, tmp); add_res_inplace(x[m], tmp, H);
        rms_prep_x(x[m], A.pre_ff.data(), H, G_I8_ACTIVE ? mlp_in[m] : nullptr, XA.xe + (size_t)m * (H / 2), XA.xo + (size_t)m * (H / 2), XA.gs + (size_t)m * (H / 32), true);
      });
      const int I = A.gate.N;
      if (!G_I8_ACTIVE) {
        XB.K = I; XB.M = M; XB.src = &act[0][0]; XB.qdone = false;
        GateUpArg gua{&XA, &A.gate, &A.up, &gate[0][0], &up[0][0], &act[0][0], &XB};
        POOL.run(gate_up_fused_task, &gua);
      } else {
        const Q4* qq[2] = {&A.gate, &A.up}; float* yy[2] = {&gate[0][0], &up[0][0]}; lin_multi(XA, 2, qq, yy);
        PF(M, [&](int m, int) { const float* gm = &gate[0][0] + (size_t)m * I; const float* um = &up[0][0] + (size_t)m * I; float* am = &act[0][0] + (size_t)m * I;
          for (int i = 0; i < I; i += 8) { __m256 g = rbf8(_mm256_loadu_ps(gm + i)), u = rbf8(_mm256_loadu_ps(um + i)); _mm256_storeu_ps(am + i, rbf8(_mm256_mul_ps(rbf8(gelu8(g)), u))); } });
        XB.prep(&act[0][0], I, M);
      }
      lin1(XB, A.down, &o[0][0]);
      XA.K = H; XA.M = M; XA.src = &x[0][0]; XA.qdone = false;
      PF(M, [&](int m, int tid) {
        float* tmp = TMP[tid]; rbf_n(o[m], H); rms(o[m], A.post_ff.data(), H, tmp);
        add_res_prep_x(x[m], tmp, H, XA.xe + (size_t)m * (H / 2), XA.xo + (size_t)m * (H / 2), XA.gs + (size_t)m * (H / 32));
      });
      if (!G_I8_ACTIVE) {
        XB.K = PD; XB.M = M; XB.src = &g2[0][0]; XB.qdone = false;
        PgFusedArg pga{&XA, &A.pg, &pgv[0][0], &g2[0][0], &pli[0][0], l, &XB};
        POOL.run(pg_fused_task, &pga);
      } else {
        lin1(XA, A.pg, &pgv[0][0]);
        PF(M, [&](int m, int) { for (int i = 0; i < PD; i += 8) { __m256 g = rbf8(_mm256_loadu_ps(&pgv[0][0] + (size_t)m * PD + i)); _mm256_storeu_ps(g2[m] + i, rbf8(_mm256_mul_ps(rbf8(gelu8(g)), rbf8(_mm256_loadu_ps(pli[m] + l * PD + i))))); } });
        XB.prep(&g2[0][0], PD, M);
      }
      lin1(XB, A.pp, &o[0][0]);
      const __m256 vsc = _mm256_set1_ps(A.scalar);
      if (l + 1 < NL) {
        XA.K = H; XA.M = M; XA.src = &a[0][0]; XA.qdone = false;
        const float* next_ln = L[l + 1].in_ln.data();
        PF(M, [&](int m, int tid) {
          float* tmp = TMP[tid]; rbf_n(o[m], H); rms(o[m], A.post_ple.data(), H, tmp);
          for (int i = 0; i < H; i += 8) _mm256_storeu_ps(x[m] + i, rbf8(_mm256_mul_ps(rbf8(_mm256_add_ps(_mm256_loadu_ps(x[m] + i), rbf8(_mm256_loadu_ps(tmp + i)))), vsc)));
          rms_prep_x(x[m], next_ln, H, G_I8_ACTIVE ? a[m] : nullptr, XA.xe + (size_t)m * (H / 2), XA.xo + (size_t)m * (H / 2), XA.gs + (size_t)m * (H / 32), false);
        });
      } else {
        PF(M, [&](int m, int tid) {
          float* tmp = TMP[tid]; rbf_n(o[m], H); rms(o[m], A.post_ple.data(), H, tmp);
          for (int i = 0; i < H; i += 8) _mm256_storeu_ps(x[m] + i, rbf8(_mm256_mul_ps(rbf8(_mm256_add_ps(_mm256_loadu_ps(x[m] + i), rbf8(_mm256_loadu_ps(tmp + i)))), vsc)));
          if (want && m >= hf) { rms(x[m], fnorm.data(), H, hbuf[m]); rbf_n(hbuf[m], H); }
        });
      }
    }
    T_LAYERS += std::chrono::duration<double>(std::chrono::steady_clock::now() - tl0).count();
    if (!want) return;
    auto th0 = std::chrono::steady_clock::now();
    auto head_group = [&](int hf, int Mend) { const int Mh = Mend - hf; auto cap = [](float s) { return r16(30.0f * tanhf(s / 30.0f)); }; bool done_screen = false;
    auto calc_thr = [&](float L) -> float {
      float Lr = r16(L), cmL = cap(Lr), lo = Lr - 64.0f, hi = Lr;
      for (int it = 0; it < 40; ++it) { float mid = 0.5f * (lo + hi); if (cap(r16(mid)) == cmL) hi = mid; else lo = mid; }
      return hi - 0.25f;
    };
    if (SCR.on && Mh <= 4) {
      static XP XH; static ScrArg SA; static float hn[4]; constexpr int LIM = 30000; static int fi[LIM]; static float fval[LIM]; static RescArg RA;
      XH.prep(&hbuf[hf][0], H, Mh); for (int m = 0; m < Mh; ++m) hn[m] = (float)sqrt(ss_avx2(hbuf[hf + m], H));
      // Fast seed lower-bound pre-screen (Lean: seed_prescreen_sound)
      float seed_L[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
      int seeds[160]; int nseeds = 0;
      for (int m = 0; m < std::min(M, 8); ++m) if (toks[m] >= 0 && toks[m] < 262144) seeds[nseeds++] = toks[m];
      if (g_sec >= 0 && g_sec < 262144) seeds[nseeds++] = g_sec;
      if (g_d_order) {
        for (int i = 0; i < 16; ++i) seeds[nseeds++] = (int)g_d_order[i];
      } else {
        const int fixed_seeds[] = {2, 106, 107, 108, 235248, 235265, 235269, 476, 573, 577, 604, 651, 688, 728, 785, 818, 947, 1024, 1154, 1280};
        for (int s : fixed_seeds) seeds[nseeds++] = s;
      }
      for (int si = 0; si < nseeds; ++si) {
        if (si + 2 < nseeds) {
          const uint8_t* Wp = SCR.w.data() + (size_t)seeds[si + 2] * (H / 2);
          for (int pf = 0; pf < H / 2; pf += 64) _mm_prefetch((const char*)(Wp + pf), _MM_HINT_T0);
        }
        int n = seeds[si]; float sd[4];
        const uint8_t* Wn = SCR.w.data() + (size_t)n * (H / 2); const uint16_t* Scn = SCR.s.data() + (size_t)n * (H / 32);
        switch (Mh) {
          case 1: w4::gemm_rows_tile<1>(Wn, Scn, XH.xe, XH.xo, XH.gs, H, 0, 1, sd, 1, H / 2, H / 32); break;
          case 2: w4::gemm_rows_tile<2>(Wn, Scn, XH.xe, XH.xo, XH.gs, H, 0, 1, sd, 1, H / 2, H / 32); break;
          case 3: w4::gemm_rows_tile<3>(Wn, Scn, XH.xe, XH.xo, XH.gs, H, 0, 1, sd, 1, H / 2, H / 32); break;
          default: w4::gemm_rows_tile<4>(Wn, Scn, XH.xe, XH.xo, XH.gs, H, 0, 1, sd, 1, H / 2, H / 32); break;
        }
        float c = SCR.coef[n];
        for (int m = 0; m < Mh; ++m) seed_L[m] = std::max(seed_L[m], sd[m] - c * hn[m]);
      }
      for (int m = 0; m < Mh; ++m) {
        SA.pre_thr[m] = seed_L[m];
        SA.shared_best[m].store(seed_L[m], std::memory_order_relaxed);
        SA.overflow[m].store(false, std::memory_order_relaxed);
      }
      SA.S = &SCR; SA.x = &XH; SA.hn = hn; SA.M = Mh; SA.nxt.store(0, std::memory_order_relaxed); POOL.run(screen_task, &SA);
      const int actT = POOL.active_count();
      float Lm[4]; bool ok = true; for (int m = 0; m < Mh; ++m) { float L = -1e30f; for (int t = 0; t < actT; ++t) L = std::max(L, SA.tmax[t][m]); Lm[m] = L; }
      auto eval_row = [&](int m) -> bool {
        if (SA.overflow[m].load(std::memory_order_relaxed)) return false;
        const float thr = calc_thr(Lm[m]);
        int tot = 0;
        for (int t = 0; t < actT; ++t) {
          const int c = SA.t_cnt[t][m];
          const int* idx_t = SA.t_idx[t][m];
          const float* u_t = SA.t_u[t][m];
          for (int k = 0; k < c; ++k) {
            if (u_t[k] >= thr) {
              if (__builtin_expect(tot >= LIM, 0)) return false;
              fi[tot++] = idx_t[k];
            }
          }
        }
        RA.P = &HP; RA.h = hbuf[hf + m]; RA.fval = fval; RA.idx = fi; RA.cnt = tot; POOL.run(resc_task, &RA);
        float mx = -1e30f; for (int t = 0; t < actT; ++t) mx = std::max(mx, RA.tmax[t]);
        float cm = cap(mx); int best = -1;
        for (int i = 0; i < tot; ++i) { float v = fval[i]; if (v >= mx - 2.0f && cap(v) == cm) { best = fi[i]; break; } }
        G[hf + m] = best; scr_rows++; scr_surv += tot; scr_max = std::max<long>(scr_max, (long)tot);
        return true;
      };
      if (hf == 0 && Mh > 1) {
        if (!eval_row(0)) ok = false;
        else if (sibling > 0 && G[0] != toks[1] && G[0] == toks[sibling]) {
          if (!eval_row(sibling)) ok = false;
        } else {
          const int chain_end = sibling > 0 ? sibling : Mh;
          for (int m = 1; m < chain_end; ++m) {
            if (G[m - 1] != toks[m]) break;
            if (!eval_row(m)) { ok = false; break; }
          }
        }
      } else {
        for (int m = 0; m < Mh; ++m) if (!eval_row(m)) { ok = false; break; }
      }
      if (ok) done_screen = true; else scr_fallback++;
    }
    if (!done_screen) {
    HA.P = &HP; HA.h = &hbuf[hf][0]; HA.logit = &logit[0][0]; HA.M = Mh; POOL.run(head_task, &HA);
    for (int m = 0; m < Mh; ++m) { float* lg = &logit[0][0] + (size_t)m * 262144; __m256 vm = _mm256_set1_ps(-1e30f);
      for (int i = 0; i < 262144; i += 8) { __m256 r = rbf8(_mm256_loadu_ps(lg + i)); _mm256_storeu_ps(lg + i, r); vm = _mm256_max_ps(vm, r); }
      float tb[8]; _mm256_storeu_ps(tb, vm); float mx = tb[0]; for (int i = 1; i < 8; ++i) mx = std::max(mx, tb[i]);
      float cm = cap(mx); int best = -1; for (int i = 0; i < 262144; ++i) if (lg[i] >= mx - 2.0f && cap(lg[i]) == cm) { best = i; break; } G[hf + m] = best;
        if (getenv("ENGINE_DEBUG_GAP")) { float t1 = -1e30f, t2 = -1e30f; int i1 = -1, i2 = -1; for (int n = 0; n < 262144; ++n) { float v = lg[n]; if (v > t1) { t2 = t1; i2 = i1; t1 = v; i1 = n; } else if (v > t2) { t2 = v; i2 = n; } } fprintf(stderr, "GAP %.4f %.4f id1=%d id2=%d chosen=%d\n", t1, t1 - t2, i1, i2, best); } }
    }
    };
    for (int r0 = hf; r0 < M; r0 += 4) head_group(r0, std::min(M, r0 + 4));
    T_HEAD += std::chrono::duration<double>(std::chrono::steady_clock::now() - th0).count();
  };


  if (G_I8) { auto t0r = std::chrono::steady_clock::now(); std::vector<std::pair<const Q4*, w4a8::Packed*>> jobs;
    auto add = [&](const Q4& q) { jobs.push_back({&q, const_cast<w4a8::Packed*>(q.p8)}); };
    for (auto& A : L) { add(A.qp); if (A.owner) { add(A.kp); add(A.vp); } add(A.op); add(A.gate); add(A.up); add(A.down); add(A.pg); add(A.pp); } add(plproj);
    struct RP { std::vector<std::pair<const Q4*, w4a8::Packed*>>* j; } rp{&jobs};
    for (auto& jb : jobs) { struct A2 { const Q4* q; w4a8::Packed* p; } a2{jb.first, jb.second};
      POOL.run([](void* pp, int tid, int T) { A2& a = *(A2*)pp; int NT = a.q->N / 8; int t0 = (long)tid * NT / T, t1 = (long)(tid + 1) * NT / T; w4a8::repack_tiles(a.q->w, a.q->s, a.q->N, a.q->K, *a.p, t0, t1); }, &a2); }
    fprintf(stderr, "int8 prefill repack: %.2fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0r).count()); }
  // ---- drafter setup ----
  int POL[3] = {-1, -1, -1}; if (const char* pp = getenv("ENGINE_POLICY")) sscanf(pp, "%d,%d,%d", &POL[0], &POL[1], &POL[2]);
  const int NG_N = getenv("ENGINE_NGRAM") ? atoi(getenv("ENGINE_NGRAM")) : 0, NG_LEN = getenv("ENGINE_NGLEN") ? std::min(7, atoi(getenv("ENGINE_NGLEN"))) : 3;
  const float TAU = getenv("ENGINE_TAU") ? (float)atof(getenv("ENGINE_TAU")) : 0.0f; const bool LOGR = getenv("ENGINE_LOG_ROUNDS") != nullptr;
  const bool SAMEPOS = getenv("ENGINE_DRAFT_SAMEPOS") && atoi(getenv("ENGINE_DRAFT_SAMEPOS")) == 1;
  int K_DRAFT = getenv("ENGINE_K") ? atoi(getenv("ENGINE_K")) : 3; bool TREE = getenv("ENGINE_TREE") && atoi(getenv("ENGINE_TREE")) == 1;
  std::vector<DLayer> DL(4); B16 d_pre, d_post, d_cent; std::vector<float> d_norm; const uint16_t* d_emb = nullptr; const int64_t* d_order = nullptr;
  if (K_DRAFT > 0) { const char* dc = getenv("ENGINE_DRAFTER_DIR") ? getenv("ENGINE_DRAFTER_DIR") : "ckpt-e2b-assistant"; const char* di = getenv("ENGINE_DRAFTER_INDEX") ? getenv("ENGINE_DRAFTER_INDEX") : "runs/drafter_index.tsv";
    { std::ifstream f(di); std::string line; while (std::getline(f, line)) { std::stringstream ss(line); std::string n, d, o, b, sh; std::getline(ss, n, '\t'); std::getline(ss, d, '\t'); std::getline(ss, o, '\t'); std::getline(ss, b, '\t'); std::getline(ss, sh, '\t');
        TInfo t; t.dtype = d; t.off = std::stoull(o); t.nbytes = std::stoull(b); std::stringstream s2(sh); std::string x; while (std::getline(s2, x, ',')) t.shape.push_back(std::stoll(x)); DIDX[n] = t; } }
    int dfd = open((std::string(dc) + "/model.safetensors").c_str(), O_RDONLY); struct stat dst; fstat(dfd, &dst); DBASE = (uint8_t*)mmap(nullptr, dst.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, dfd, 0);
    for (int l = 0; l < 4; ++l) { DLayer& d = DL[l]; std::string p = "model.layers." + std::to_string(l) + "."; d.sliding = l < 3; d.hd = d.sliding ? 256 : 512;
      d.qp = db16(p + "self_attn.q_proj.weight"); d.op = db16(p + "self_attn.o_proj.weight"); d.gate = db16(p + "mlp.gate_proj.weight"); d.up = db16(p + "mlp.up_proj.weight"); d.down = db16(p + "mlp.down_proj.weight");
      d.in_ln = dvec(p + "input_layernorm.weight"); d.post_attn = dvec(p + "post_attention_layernorm.weight"); d.pre_ff = dvec(p + "pre_feedforward_layernorm.weight"); d.post_ff = dvec(p + "post_feedforward_layernorm.weight");
      d.qn = dvec(p + "self_attn.q_norm.weight"); d.scalar = bf(dbf(p + "layer_scalar")[0]); }
    d_pre = db16("pre_projection.weight"); d_post = db16("post_projection.weight"); d_cent = db16("masked_embedding.centroids.weight"); d_norm = dvec("model.norm.weight");
    d_emb = dbf("model.embed_tokens.weight"); d_order = (const int64_t*)(DBASE + DIDX["masked_embedding.token_ordering"].off); g_d_order = d_order; }
  static float dcomb[3072], dx[256], dres[256], da[256], dq[2048], dopt[2048], dtmp[512], dgate[2048], dup[2048], dact[2048], dscore[2048], dbk[1536], dsc[40960];
  static float dh[256]; static int cand[4096]; static float cscore[4096]; static float last_p = 0;
  // one draft step: (token, backbone hidden h[1536], query position pos, last valid target position P) -> next token; writes b_out[1536]
  static const bool DRAFT_SAME = !(getenv("ENGINE_DRAFT_SAME") && atoi(getenv("ENGINE_DRAFT_SAME")) == 0);   // drafter keeps the forward's thread count (no park/wake churn at long context)
  static const int DRAFT_CUT = getenv("ENGINE_THREAD_CUTOFF") ? atoi(getenv("ENGINE_THREAD_CUTOFF")) : 192;
  auto draft_step = [&](int tok, const float* h, int pos, int P, float* b_out) -> int {
    POOL.set_active(DRAFT_SAME && getenv("ENGINE_ADAPTIVE_THREADS") && P + 1 > DRAFT_CUT ? 8 : 6);
    const uint16_t* er = E + (size_t)tok * H; const __m256 vsqh = _mm256_set1_ps(sqrtf((float)H));
    for (int i = 0; i < H; i += 8) _mm256_storeu_ps(dcomb + i, rbf8(_mm256_mul_ps(bf8(er + i), vsqh)));
    std::memcpy(dcomb + H, h, H * 4);
    b16_1(dcomb, d_pre, dx); rbf_n(dx, 256);
    for (int l = 0; l < 4; ++l) { DLayer& D = DL[l]; const int hd = D.hd, ty = D.sliding ? 0 : 1, ol = D.sliding ? 13 : 14; const float* cs = &RC[ty][(size_t)pos * hd / 2]; const float* sn = &RS[ty][(size_t)pos * hd / 2];
      rms(dx, D.in_ln.data(), 256, da); b16_1(da, D.qp, dq);
      for (int h2 = 0; h2 < 4; ++h2) { float* qh = dq + h2 * hd; rms_r16_inplace(qh, D.qn.data(), hd); rope_avx2(qh, cs, sn, hd); }
      { AttnArg aa; aa.Kc = KC[ol].data(); aa.Vc = VC[ol].data(); aa.hd = hd; aa.nh = 4; aa.M = 1; aa.q[0] = dq; aa.pos[0] = P; aa.skip_lo[0] = 1; aa.skip_hi[0] = 0; aa.lo[0] = D.sliding ? std::max(0, pos - 511) : 0; aa.kmin = aa.lo[0]; aa.kmax = P;
        float* outp[1] = {dopt}; POOL.run(attn_task, &aa); attn_merge(aa, POOL.active_count(), outp); }
      b16_1(dopt, D.op, da); rbf_n(da, 256); rms(da, D.post_attn.data(), 256, dtmp); add_res_inplace(dx, dtmp, 256);
      rms(dx, D.pre_ff.data(), 256, da); rbf_n(da, 256);
      { DGateUpArg dgua{da, &D.gate, &D.up, dgate, dup, dact}; POOL.run(d_gate_up_task, &dgua); }
      b16_1(dact, D.down, da); rbf_n(da, 256); rms(da, D.post_ff.data(), 256, dtmp); add_res_inplace(dx, dtmp, 256);
      __m256 vdsc = _mm256_set1_ps(D.scalar);
      for (int i = 0; i < 256; i += 8) _mm256_storeu_ps(dx + i, rbf8(_mm256_mul_ps(_mm256_loadu_ps(dx + i), vdsc))); }
    rms(dx, d_norm.data(), 256, dh); rbf_n(dh, 256);
    { const B16* mm[2] = {&d_post, &d_cent}; float* yy[2] = {b_out, dscore}; b16_multi(dh, 2, mm, yy); }
    rbf_n(b_out, H);
    int top[32]; { float thr[32]; int cnt = 0; for (int c = 0; c < 2048; ++c) { float s = dscore[c]; if (cnt < 32) { int j = cnt++; while (j > 0 && thr[j - 1] < s) { thr[j] = thr[j - 1]; top[j] = top[j - 1]; --j; } thr[j] = s; top[j] = c; }
        else if (s > thr[31]) { int j = 31; while (j > 0 && thr[j - 1] < s) { thr[j] = thr[j - 1]; top[j] = top[j - 1]; --j; } thr[j] = s; top[j] = c; } } }
    int nc = 0; for (int t = 0; t < 32; ++t) for (int j = 0; j < 128; ++j) cand[nc++] = (int)d_order[(size_t)top[t] * 128 + j];
    { DCandArg dca{d_emb, cand, dh, cscore, nc}; POOL.run(dcand_task, &dca); }
    int bestc = cand[0]; float bs = -1e30f; float bs2 = -1e30f; int sec = -1;
    for (int ci = 0; ci < nc; ++ci) { float sc2 = cscore[ci]; if (sc2 > bs) { bs2 = bs; sec = bestc; bs = sc2; bestc = cand[ci]; } else if (sc2 > bs2) { bs2 = sc2; sec = cand[ci]; } }
    g_sec = sec; g_gap = bs - bs2;
    if (TAU > 0.0f) { double den = 0; for (int ci = 0; ci < nc; ++ci) den += exp((double)cscore[ci] - bs); last_p = (float)(1.0 / den); } else last_p = 1.0f;
    return bestc;
  };

  AudioTower AT; if (getenv("ENGINE_AUDIO_TOWER")) { auto ta = std::chrono::steady_clock::now(); AT.load(); fprintf(stderr, "audio tower loaded: %.2fs\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count()); }
  const int AUDIO_ID = 258881; std::vector<float> AUD; int AUD_ROWS = 0;
  if (const char* ap = getenv("ENGINE_AUDIO_EMB")) { FILE* af = fopen(ap, "rb"); fseek(af, 0, SEEK_END); long sz = ftell(af); fseek(af, 0, SEEK_SET); AUD.resize(sz / 4); if (fread(AUD.data(), 4, AUD.size(), af) != AUD.size()) {} fclose(af); AUD_ROWS = (int)(AUD.size() / H); fprintf(stderr, "audio embeddings: %d rows\n", AUD_ROWS); }
  // ---- request runner with prompt (prefix) cache: KV at positions < kv_len is valid for ctx_tokens ----
  std::vector<uint64_t> ctx_keys; int kv_len = 0;
  // Multi-slot prefix cache: the active KV (KC/VC/ctx_keys/kv_len) plus NSLOT-1 stored conversations. A request runs on the slot with the longest shared prefix;
  // if nothing matches, the least-recently-used stored slot is sacrificed and the previously active conversation is parked in its place (never dropped).
  struct Slot { std::vector<std::vector<uint16_t>> KC, VC; std::vector<uint64_t> keys; int kv_len = 0; uint64_t stamp = 0; };
  const int NSLOT = std::max(1, getenv("ENGINE_SLOTS") ? atoi(getenv("ENGINE_SLOTS")) : 1); const int MIN_REUSE = getenv("ENGINE_SLOT_MIN_REUSE") ? atoi(getenv("ENGINE_SLOT_MIN_REUSE")) : 32;
  std::vector<Slot> stored(NSLOT - 1); uint64_t clock_tick = 0, active_stamp = 0; int slot_switches = 0, slot_evictions = 0;
  for (auto& sl : stored) { sl.KC.resize(15); sl.VC.resize(15); for (int l = 0; l < 15; ++l) { sl.KC[l].assign((size_t)ctxmax * L[l].hd, 0); sl.VC[l].assign((size_t)ctxmax * L[l].hd, 0); } }
  auto swap_with = [&](Slot& sl) { std::swap(KC, sl.KC); std::swap(VC, sl.VC); std::swap(ctx_keys, sl.keys); std::swap(kv_len, sl.kv_len); std::swap(active_stamp, sl.stamp); };
  { int w = 2; fwd(&w, 1, 0, true); }
  if (getenv("ENGINE_MLOCK")) lock_hot_anon();
  if (getenv("ENGINE_KEEPWARM")) { std::vector<std::pair<const uint8_t*, size_t>> hot; for (auto& kv : IDX) if (kv.first.find("layers.") != std::string::npos && (kv.first.find("weight_packed") != std::string::npos || kv.first.find("weight_scale") != std::string::npos)) hot.push_back({BASE + kv.second.off, kv.second.nbytes});
    std::thread(keepwarm_loop, std::move(hot), cpus).detach(); }   // one-time warm-up (code, caches, predictors); kv_len stays 0
  auto row_hash = [&](int r) { uint64_t h = 1469598103934665603ull; const unsigned char* p = (const unsigned char*)&AUD[(size_t)r * H]; for (int i = 0; i < H * 4; ++i) { h ^= p[i]; h *= 1099511628211ull; } return (1ull << 40) | (h & 0xFFFFFFFFFFull); };
  auto run_req = [&](const std::vector<int>& prompt, int nnew_req, bool print) {
    tree_attempts = 0; tree_hits = 0;
    const int n = (int)prompt.size(); std::vector<uint64_t> keys(n); { int ar = 0; for (int i = 0; i < n; ++i) keys[i] = (prompt[i] == AUDIO_ID && ar < AUD_ROWS) ? row_hash(ar++) : (uint64_t)prompt[i]; }
    auto lcp_of = [&](const std::vector<uint64_t>& k, int klen) { int c = 0; while (c < klen && c < n && k[c] == keys[c]) ++c; return c; };
    int common = lcp_of(ctx_keys, kv_len); ++clock_tick; int slot_used = 0;
    if (NSLOT > 1) {
      int best = -1, bc = common; for (int i = 0; i < (int)stored.size(); ++i) { const int c = lcp_of(stored[i].keys, stored[i].kv_len); if (c > bc) { bc = c; best = i; } }
      if (best >= 0) { active_stamp = clock_tick; swap_with(stored[best]); ++slot_switches; slot_used = best + 1; common = bc; }                  // a stored conversation shares more of this prompt: switch to it (current one is parked in its place)
      else if (bc < MIN_REUSE) { int v = 0; for (int i = 1; i < (int)stored.size(); ++i) if (stored[i].stamp < stored[v].stamp) v = i;                     // nothing worth reusing: sacrifice the LRU slot, park the current conversation there
        active_stamp = clock_tick; swap_with(stored[v]); kv_len = 0; ctx_keys.clear(); common = 0; ++slot_evictions; slot_used = v + 1; }
      active_stamp = clock_tick;
    }
    const int start = std::min(common, n - 1);   // always re-run the last prompt token to obtain its logits/hidden
    auto T0 = std::chrono::steady_clock::now(); int done = start; G_I8_ACTIVE = G_I8;
    int aud_before = 0; for (int i = 0; i < start; ++i) if (prompt[i] == AUDIO_ID) ++aud_before;
    while (done < n) { int M = std::min(PFB, n - done); bool last = done + M == n; const float* ep[32]; bool anyaud = false;
      for (int m = 0; m < M; ++m) { if (prompt[done + m] == AUDIO_ID && aud_before < AUD_ROWS) { ep[m] = &AUD[(size_t)aud_before * H]; ++aud_before; anyaud = true; } else ep[m] = nullptr; }
      fwd(prompt.data() + done, M, FAKE_N + done, last, last ? M - 1 : 0, anyaud ? ep : nullptr); done += M; }
    { static const bool DEC_I8 = !(getenv("ENGINE_DECODE_INT8") && atoi(getenv("ENGINE_DECODE_INT8")) == 0);  /* default ON; =0 for exact float */ G_I8_ACTIVE = DEC_I8 && G_I8; } double tp = std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
    const int lastrow = (n - start - 1) % PFB; std::vector<int> gen; gen.push_back(G[lastrow]); auto T1 = std::chrono::steady_clock::now(); 
#ifdef PROF_NM
 NPROF = 0; for (auto& e : PROFV) e = ProfE{}; unsigned long long tsc0 = __rdtsc(); auto wall0 = std::chrono::steady_clock::now();
#endif
 const double layers0 = T_LAYERS, att0 = T_ATT, head0 = T_HEAD, draft0 = T_DRAFT; const long fwd0 = N_FWD;
    if (print && nnew_req > 0) { printf("FIRST %d\n", G[lastrow]); fflush(stdout); }
    if (nnew_req <= 0) { ctx_keys.assign(keys.begin(), keys.end()); kv_len = n; if (print) { printf("GEN:\nTIMING prompt=%d reused=%d new=%d prefill_s=%.3f decode_tokens=0 decode_s=0 decode_tok_s=0 tokens_per_round=0 ngram_rounds=0/0 ngram_acc_per_round=0\n", n, start, n - start, tp); fflush(stdout); } return; }
    long rounds = 0, acc_pos[4] = {0, 0, 0, 0}; int P; std::vector<int> seq; seq.reserve(8192); long ng_rounds = 0, ng_acc = 0;
    auto is_stop = [](int t) { return t == 106 || t == 1; };
    static const bool STREAM = getenv("ENGINE_STREAM") != nullptr;   // serve mode: announce committed tokens after every round ("TOK id id ..."), FIRST already announced gen[0]
    size_t emitted = 1;
    auto emit = [&]() { if (!STREAM || !print) return; const size_t lim = std::min<size_t>(gen.size(), (size_t)nnew_req); std::string o; size_t i = emitted;
      for (; i < lim; ++i) { o += ' '; o += std::to_string(gen[i]); if (is_stop(gen[i])) { ++i; break; } }
      if (i > emitted) { printf("TOK%s\n", o.c_str()); fflush(stdout); emitted = is_stop(gen[i - 1]) ? lim : i; } };
    if (K_DRAFT == 0) { for (int i = 1; i < nnew_req && !is_stop(gen.back()); ++i) { int t = gen.back(); fwd(&t, 1, FAKE_N + n + i - 1, true); gen.push_back(G[0]); emit(); } P = FAKE_N + n + std::max(0, nnew_req - 1) - 1; }
    else {
      static float hprev[1536], bcur[1536], hcur[1536]; std::memcpy(hprev, hbuf[lastrow], H * 4); P = FAKE_N + n - 1; int conf = gen.back(); int state = 1;
      bool stopped = is_stop(gen.back());
      while ((int)gen.size() < nnew_req && !stopped) {
        emit();
        int block[16]; block[0] = conf; std::memcpy(hcur, hprev, H * 4); int cur = conf; int k = std::min(K_DRAFT, 3); if (POL[0] >= 0) k = std::min(k, POL[state]);
        if (TREE) k = std::min(k, 2);
        int nd = 0; float cum = 1.0f; float pj[3] = {0, 0, 0}; bool used_ng = false;
        if (NG_N > 0) { // prompt-lookup: most recent earlier occurrence of the last NG_N tokens of (prompt + generated), copy what followed
          seq.assign(prompt.begin(), prompt.end()); seq.insert(seq.end(), gen.begin(), gen.end()); const int L = (int)seq.size();
          if (L > NG_N) for (int i = L - NG_N - 1; i >= 0; --i) { bool eq = true; for (int t = 0; t < NG_N; ++t) if (seq[i + t] != seq[L - NG_N + t]) { eq = false; break; }
            if (eq) { for (int j = 0; j < NG_LEN && i + NG_N + j < L; ++j) block[1 + nd++] = seq[i + NG_N + j]; break; } }
          if (nd > 0) { used_ng = true; ++ng_rounds; } }
        static const float SIB_GAP = getenv("ENGINE_SIB_GAP") ? (float)atof(getenv("ENGINE_SIB_GAP")) : 1.0f;
        int sec1 = -1; float gap1 = -1.0f; bool use_sib_m3 = false;
        auto tdr0 = std::chrono::steady_clock::now();
        for (int j = 0; j < (used_ng ? 0 : k); ++j) {
          int d = draft_step(cur, hcur, SAMEPOS ? P + 1 : P + 1 + j, P, bcur);
          pj[j] = last_p;
          if (j == 0) {
            sec1 = g_sec; gap1 = g_gap;
            const bool allow_sib = (SIB_GAP >= 0.0f && rounds >= 1 && state < 2 && k >= 2 && sec1 >= 0 && sec1 != d);
            static const bool SIB_LONG = getenv("ENGINE_SIB_LONG") != nullptr;   // in split-KV regime the sibling is never committed (q2=0), so the chain is optimal (Lean sibling_beats_chain_iff)
            const bool sib_cond = (P + 1 + k > ATT_PAR) ? (SIB_LONG && (g_gap <= 0.85f || (g_gap >= 1.65f && g_gap <= 2.55f))) : (g_gap <= ((rounds <= 3 || state == 1) ? 2.55f : 1.80f));
            if (allow_sib && sib_cond) {
              block[1] = d; ++nd; use_sib_m3 = true; break;
            }
          }
          cum *= last_p; if (cum < TAU) break; block[1 + j] = d; ++nd; cur = d; std::memcpy(hcur, bcur, H * 4);
        }
        T_DRAFT += std::chrono::duration<double>(std::chrono::steady_clock::now() - tdr0).count();
        const int sibling = ((TREE || use_sib_m3) && !used_ng && nd >= 1 && sec1 >= 0 && sec1 != block[1]) ? nd + 1 : -1;
        if (sibling >= 0) { block[sibling] = sec1; ++tree_attempts; }
        fwd(block, nd + 1 + (sibling >= 0), P + 1, true, 0, nullptr, sibling); ++rounds;
        if (sibling >= 0 && (TREE || (P + 1 + sibling <= ATT_PAR && gap1 >= 0.25f)) && block[1] != G[0] && sec1 == G[0]) { ++tree_hits;
          for (int l = 0; l < 15; ++l) { const int hd = L[l].hd; std::memcpy(KC[l].data() + (size_t)(P + 2) * hd, KC[l].data() + (size_t)(P + 1 + sibling) * hd, hd * sizeof(uint16_t)); std::memcpy(VC[l].data() + (size_t)(P + 2) * hd, VC[l].data() + (size_t)(P + 1 + sibling) * hd, hd * sizeof(uint16_t)); }
          gen.push_back(G[0]); if (!is_stop(G[0]) && (int)gen.size() < nnew_req) gen.push_back(G[sibling]);
          stopped = is_stop(gen.back()); conf = G[sibling]; std::memcpy(hprev, hbuf[sibling], H * 4); P += 2; state = 1; continue;
        }
        if (!used_ng && nd >= 1) { ++R1_n; if (block[1] == G[0]) ++R1_top1; else if (sec1 == G[0]) ++R1_top2; }
        int nacc = 0; while (nacc < nd && block[nacc + 1] == G[nacc]) { ++nacc; if (nacc <= 3) ++acc_pos[nacc - 1]; }
        if (used_ng) ng_acc += nacc;
        state = (nacc == 0) ? 0 : ((!use_sib_m3 && nacc == nd) ? 2 : 1);
        for (int i = 0; i <= nacc; ++i) { gen.push_back(G[i]); if (is_stop(G[i])) { stopped = true; break; } }
        conf = G[nacc]; std::memcpy(hprev, hbuf[nacc], H * 4); P += nacc + 1;
      }
    }
    emit();
    double td = std::chrono::duration<double>(std::chrono::steady_clock::now() - T1).count();
    
#ifdef PROF_NM
    { double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count(); double hz = (__rdtsc() - tsc0) / wall; unsigned long long ta = 0, tw = 0;
      for (int i = 0; i < NPROF; ++i) { ta += PROFV[i].own; tw += PROFV[i].wait;
        fprintf(stderr, "PROF off=%lx n=%llu own_ms=%.2f wait_ms=%.2f per_call_us=%.2f\n", (unsigned long)((char*)PROFV[i].f - (char*)&main), PROFV[i].n, PROFV[i].own * 1e3 / hz, PROFV[i].wait * 1e3 / hz, (PROFV[i].own + PROFV[i].wait) * 1e6 / hz / PROFV[i].n); }
      fprintf(stderr, "PROF total_in_run own_ms=%.1f wait_ms=%.1f\n", ta * 1e3 / hz, tw * 1e3 / hz); }
#endif
    if (getenv("ENGINE_DECODE_STAGE")) fprintf(stderr, "DECODE_STAGE forwards=%ld layers_ms=%.1f split_att_ms=%.1f head_ms=%.1f draft_ms=%.1f total_ms=%.1f\n", N_FWD - fwd0, (T_LAYERS-layers0)*1000, (T_ATT-att0)*1000, (T_HEAD-head0)*1000, (T_DRAFT-draft0)*1000, td*1000);
    if (print) fprintf(stderr, "RANK1 rounds=%ld top1=%ld top2_only=%ld TREE attempts=%ld hits=%ld\n", R1_n, R1_top1, R1_top2, tree_attempts, tree_hits);
    // KV is valid for positions 0..P: prompt, then generated tokens gen[0..P-n]
    ctx_keys.assign(keys.begin(), keys.end()); for (int i = 0; i <= P - FAKE_N - n && i < (int)gen.size(); ++i) ctx_keys.push_back((uint64_t)gen[i]); kv_len = FAKE_N ? 0 : std::min<int>(P + 1, (int)ctx_keys.size());
    if ((int)gen.size() > nnew_req) gen.resize(nnew_req);
    { size_t cut = gen.size(); for (size_t i = 0; i < gen.size(); ++i) if (is_stop(gen[i])) { cut = i + 1; break; } gen.resize(cut); }
    if (print) { printf("GEN:"); for (int g : gen) printf(" %d", g); printf("\n");
      { const int produced = std::max(0, (int)gen.size() - 1); printf("TIMING prompt=%d reused=%d new=%d prefill_s=%.3f decode_tokens=%d decode_s=%.3f decode_tok_s=%.2f tokens_per_round=%.2f ngram_rounds=%ld/%ld ngram_acc_per_round=%.2f slot=%d\n", n, start, n - start, tp, produced, td, produced / td, rounds ? (double)produced / rounds : 1.0, ng_rounds, rounds, ng_rounds ? (double)ng_acc / ng_rounds : 0.0, slot_used); } fflush(stdout); }
  };
  if (getenv("ENGINE_SERVE")) { std::string line; while (std::getline(std::cin, line)) {
      g_last_req_ns.store((long long)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
      if (line.rfind("BENCH_MODE ", 0) == 0) { std::stringstream ms(line); std::string cmd; int k = -1, tree = -1; ms >> cmd >> k >> tree; if (k >= 1 && k <= 3 && tree >= 0 && tree <= 1 && DBASE) { K_DRAFT = k; TREE = tree; printf("MODE k=%d tree=%d\n", k, tree); } else printf("MODE_ERROR\n"); fflush(stdout); continue; }
      if (line.rfind("AUDIO_RESET", 0) == 0) { AT.reset(); AUD.clear(); AUD_ROWS = 0; printf("OK\n"); fflush(stdout); continue; }
      if (line.rfind("AUDIO_FEED", 0) == 0) { std::stringstream fs(line); std::string cmd, path; int fin = 0; fs >> cmd >> fin >> path; std::vector<float> mm; FILE* mf = fopen(path.c_str(), "rb"); if (mf) { fseek(mf, 0, SEEK_END); long sz = ftell(mf); fseek(mf, 0, SEEK_SET); mm.resize(sz / 4); if (fread(mm.data(), 4, mm.size(), mf) != mm.size()) {} fclose(mf); }
        auto t0a = std::chrono::steady_clock::now(); auto rows = AT.feed(mm.data(), (int)(mm.size() / 128), fin != 0); double msa = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0a).count() * 1e3;
        AUD.insert(AUD.end(), rows.begin(), rows.end()); AUD_ROWS = (int)(AUD.size() / H); printf("AROWS %d new=%d ms=%.1f\n", AUD_ROWS, (int)(rows.size() / H), msa); fflush(stdout); continue; }
      if (line.rfind("AUDIO_DUMP", 0) == 0) { std::stringstream fs(line); std::string cmd, path; fs >> cmd >> path; FILE* df = fopen(path.c_str(), "wb"); fwrite(AUD.data(), 4, AUD.size(), df); fclose(df); printf("OK\n"); fflush(stdout); continue; }
      std::stringstream ss(line); int nn; if (!(ss >> nn)) continue; std::vector<int> pr; std::string w; while (ss >> w) { if (w == "A=@") { /* use the audio rows already produced by AUDIO_FEED */ } else if (w.rfind("A=", 0) == 0) { FILE* af = fopen(w.c_str() + 2, "rb"); if (af) { fseek(af, 0, SEEK_END); long sz = ftell(af); fseek(af, 0, SEEK_SET); AUD.resize(sz / 4); if (fread(AUD.data(), 4, AUD.size(), af) != AUD.size()) {} fclose(af); AUD_ROWS = (int)(AUD.size() / H); } } else pr.push_back(atoi(w.c_str())); } if (pr.empty()) continue; run_req(pr, nn, true); } }
  else { run_req(prompt, nnew, true); if (SCR.on && scr_rows) fprintf(stderr, "SCREEN rows=%ld survivors mean=%.1f max=%ld fallbacks=%ld\n", scr_rows, (double)scr_surv / scr_rows, scr_max, scr_fallback); }
  if (getenv("ENGINE_STAGE_PROF")) fprintf(stderr, "STAGES total over %ld fwd calls (ms): layers %.0f (of which split-KV attention %.0f) | head %.0f | drafter %.0f\n", N_FWD, T_LAYERS * 1e3, T_ATT * 1e3, T_HEAD * 1e3, T_DRAFT * 1e3);
  fflush(stdout); fflush(stderr); _exit(0);
}
