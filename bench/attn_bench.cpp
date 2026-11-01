// Standalone attention study: engine attn_task vs attn_task_pf (software prefetch), interleaved ABAB, cold = cycle KV copies + 8 MB scribble.
#define main engine_main
#include "../engine/gemma4_engine.cpp"
#undef main
#include <random>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static unsigned long long fnv(const std::vector<float>& v) { unsigned long long h = 1469598103934665603ULL; for (float x : v) { uint32_t u; memcpy(&u, &x, 4); h = (h ^ u) * 1099511628211ULL; } return h; }
int main() {
  std::vector<int> cpus = {0, 2, 4, 6, 8, 10, 12, 14}; POOL.start(cpus);
  const int hd = 512, len = 6800, M = 3, NC = 8; const size_t per = (size_t)len * hd;
  std::vector<std::vector<uint16_t>> Kc(NC, std::vector<uint16_t>(per)), Vc(NC, std::vector<uint16_t>(per));
  std::mt19937 g(1); for (int c = 0; c < NC; ++c) for (size_t i = 0; i < per; ++i) { Kc[c][i] = 0x3c00 + (g() & 0x3ff) - 0x200; Vc[c][i] = 0x3c00 + (g() & 0x3ff) - 0x200; }
  std::vector<float> q(M * 8 * hd); for (auto& x : q) x = ((g() & 1023) - 512) / 4096.f;
  std::vector<float> o0(M * 8 * hd), o1(M * 8 * hd);
  auto call = [&](bool pf, int c, std::vector<float>& out) {
    AttnArg aa; aa.Kc = Kc[c].data(); aa.Vc = Vc[c].data(); aa.hd = hd; aa.nh = 8; aa.M = M; aa.kmax = len - 1; aa.kmin = 0; float* outp[4];
    for (int m = 0; m < M; ++m) { aa.q[m] = q.data() + (size_t)m * 8 * hd; aa.pos[m] = len - M + m; aa.lo[m] = 0; aa.skip_lo[m] = 1; aa.skip_hi[m] = 0; outp[m] = out.data() + (size_t)m * 8 * hd; }
    double t0 = now(); POOL.run(pf ? attn_task_pf : attn_task, &aa); attn_merge(aa, POOL.active_count(), outp, nullptr); return now() - t0; };
  call(false, 0, o0); call(true, 0, o1); printf("bit-exact vs original: %s (hash %016llx / %016llx)\n", fnv(o0) == fnv(o1) ? "YES" : "NO", fnv(o0), fnv(o1));
  double t[2] = {0, 0}; std::vector<double> s[2]; const int iters = 400;
  for (int it = 0; it < iters; ++it) for (int pf = 0; pf < 2; ++pf) {
    std::vector<float> junk(1 << 21); for (size_t i = 0; i < junk.size(); i += 16) junk[i] = 1;
    double dt = call(pf == 1, (it * 2 + pf) % NC, pf ? o1 : o0); if (it >= 20) s[pf].push_back(dt);
  }
  for (int pf = 0; pf < 2; ++pf) { std::sort(s[pf].begin(), s[pf].end()); printf("%-10s cold  median %.3f ms  p10 %.3f  p90 %.3f  (hd=512, 6.8k, M=3, PD=%d)\n", pf ? "prefetch" : "original", s[pf][s[pf].size() / 2] * 1e3, s[pf][s[pf].size() / 10] * 1e3, s[pf][s[pf].size() * 9 / 10] * 1e3, ATT_PD); }
  
  // sliding-window layers: hd=256, window 512 at pos 6800; T = 8 threads; hot (same KV, as for shared layers) and cold
  { const int shd = 256; std::vector<float> sq(M * 8 * shd), so(M * 8 * shd); for (auto& x : sq) x = ((g() & 1023) - 512) / 4096.f;
    for (int cold = 0; cold < 2; ++cold) { std::vector<double> v;
      for (int it = 0; it < 300; ++it) { if (cold) { std::vector<float> junk(1 << 21); for (size_t i = 0; i < junk.size(); i += 16) junk[i] = 1; }
        const int c = cold ? it % NC : 0; AttnArg aa; aa.Kc = Kc[c].data(); aa.Vc = Vc[c].data(); aa.hd = shd; aa.nh = 8; aa.M = M; aa.kmax = len - 1; float* outp[4]; aa.kmin = len;
        for (int m = 0; m < M; ++m) { aa.q[m] = sq.data() + (size_t)m * 8 * shd; aa.pos[m] = len - M + m; aa.lo[m] = std::max(0, aa.pos[m] - 511); aa.skip_lo[m] = 1; aa.skip_hi[m] = 0; aa.kmin = std::min(aa.kmin, aa.lo[m]); outp[m] = so.data() + (size_t)m * 8 * shd; }
        double t0 = now(); POOL.run(attn_task, &aa); attn_merge(aa, POOL.active_count(), outp, nullptr); double dt = now() - t0; if (it >= 20) v.push_back(dt); }
      std::sort(v.begin(), v.end()); printf("sliding hd=256 window=512 %-5s median %.1f us  p10 %.1f  p90 %.1f   x28 layers = %.2f ms\n", cold ? "cold" : "hot", v[v.size() / 2] * 1e6, v[v.size() / 10] * 1e6, v[v.size() * 9 / 10] * 1e6, v[v.size() / 2] * 28e3); } }
  { std::vector<double> v; for (int it = 0; it < 300; ++it) { double t0 = now(); POOL.run([](void*, int, int) {}, nullptr); v.push_back(now() - t0); } std::sort(v.begin(), v.end()); printf("empty pool dispatch median %.2f us (8 threads)\n", v[v.size() / 2] * 1e6); }
  POOL.shutdown(); return 0;
}
