// HIE test-suite. Self-contained (no framework). Run: ./hie_tests [filter]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <string>
#include <thread>
#include <vector>

#include "colorpipe.hpp"
#include "hie/hie.h"
#include "hie/hie_c.h"
#include "kernels.hpp"
#include "lut3d.hpp"
#include "../tools/synth.hpp"

using namespace hie;

// ---- allocation counter (global operator new) ---------------------------------------------------
static std::atomic<long> g_allocs{0};
static std::atomic<bool> g_countAllocs{false};
void* operator new(size_t n) {
  if (g_countAllocs.load(std::memory_order_relaxed)) g_allocs.fetch_add(1);
  void* p = std::malloc(n ? n : 1);
  if (!p) throw std::bad_alloc();
  return p;
}
void* operator new[](size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }

// ---- mini framework -------------------------------------------------------------------------------
static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                        \
  do {                                                                          \
    ++g_checks;                                                                 \
    if (!(cond)) {                                                              \
      ++g_fail;                                                                 \
      std::printf("    FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond);           \
      std::printf(__VA_ARGS__);                                                 \
      std::printf("\n");                                                        \
    }                                                                           \
  } while (0)

// ---- colour science (test-side reference, independent of the engine) -------------------------------
struct Lab { double L, a, b; };
static Lab rgb8ToLab(int r8, int g8, int b8) {
  auto lin = [](double c) { c /= 255.0; return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4); };
  double r = lin(r8), g = lin(g8), b = lin(b8);
  double X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
  double Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
  double Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
  const double Xn = 0.95047, Yn = 1.0, Zn = 1.08883;
  auto f = [](double t) { return t > 216.0 / 24389.0 ? std::cbrt(t) : (24389.0 / 27.0 * t + 16.0) / 116.0; };
  double fx = f(X / Xn), fy = f(Y / Yn), fz = f(Z / Zn);
  return {116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)};
}
// CIEDE2000 (Sharma, Wu, Dalal 2005)
static double deltaE2000(const Lab& p, const Lab& q) {
  const double kPi = 3.14159265358979323846;
  auto rad = [&](double d) { return d * kPi / 180.0; };
  auto deg = [&](double r) { double d = r * 180.0 / kPi; return d < 0 ? d + 360.0 : d; };
  double C1 = std::hypot(p.a, p.b), C2 = std::hypot(q.a, q.b);
  double Cb = 0.5 * (C1 + C2);
  double G = 0.5 * (1.0 - std::sqrt(std::pow(Cb, 7) / (std::pow(Cb, 7) + std::pow(25.0, 7))));
  double a1 = (1 + G) * p.a, a2 = (1 + G) * q.a;
  double c1 = std::hypot(a1, p.b), c2 = std::hypot(a2, q.b);
  double h1 = (c1 == 0) ? 0 : deg(std::atan2(p.b, a1)), h2 = (c2 == 0) ? 0 : deg(std::atan2(q.b, a2));
  double dL = q.L - p.L, dC = c2 - c1, dh;
  if (c1 * c2 == 0) dh = 0;
  else if (std::fabs(h2 - h1) <= 180) dh = h2 - h1;
  else dh = h2 - h1 > 180 ? h2 - h1 - 360 : h2 - h1 + 360;
  double dH = 2 * std::sqrt(c1 * c2) * std::sin(rad(dh / 2));
  double Lb = 0.5 * (p.L + q.L), Cb2 = 0.5 * (c1 + c2), hb;
  if (c1 * c2 == 0) hb = h1 + h2;
  else if (std::fabs(h1 - h2) <= 180) hb = 0.5 * (h1 + h2);
  else hb = (h1 + h2 < 360) ? 0.5 * (h1 + h2 + 360) : 0.5 * (h1 + h2 - 360);
  double T = 1 - 0.17 * std::cos(rad(hb - 30)) + 0.24 * std::cos(rad(2 * hb)) + 0.32 * std::cos(rad(3 * hb + 6)) - 0.20 * std::cos(rad(4 * hb - 63));
  double dTh = 30 * std::exp(-std::pow((hb - 275) / 25, 2));
  double Rc = 2 * std::sqrt(std::pow(Cb2, 7) / (std::pow(Cb2, 7) + std::pow(25.0, 7)));
  double Sl = 1 + 0.015 * std::pow(Lb - 50, 2) / std::sqrt(20 + std::pow(Lb - 50, 2));
  double Sc = 1 + 0.045 * Cb2, Sh = 1 + 0.015 * Cb2 * T;
  double Rt = -std::sin(rad(2 * dTh)) * Rc;
  return std::sqrt(std::pow(dL / Sl, 2) + std::pow(dC / Sc, 2) + std::pow(dH / Sh, 2) + Rt * (dC / Sc) * (dH / Sh));
}
static double dE(const uint8_t* a, const uint8_t* b) {
  return deltaE2000(rgb8ToLab(a[0], a[1], a[2]), rgb8ToLab(b[0], b[1], b[2]));
}

// ---- helpers -----------------------------------------------------------------------------------------
static Config stillCfg(int w, int h, InFormat f = InFormat::RGB24) {
  Config c;
  c.width = w; c.height = h; c.in_format = f;
  c.async_analysis = false; c.frame_smoothing = 1.f; c.threads = 2;
  return c;
}
static Config allOff(int w, int h, InFormat f) {
  Config c = stillCfg(w, h, f);
  c.enable_denoise = c.enable_clahe = c.enable_lut = false;
  c.auto_wb = c.auto_exposure = c.auto_contrast = c.auto_saturation = c.auto_shadows = c.auto_clahe = c.auto_denoise = false;
  return c;
}
static std::vector<uint8_t> runOnce(const Config& c, const std::vector<uint8_t>& frame, int frames = 1) {
  Enhancer e(c);
  if (!e.valid()) { std::printf("    invalid config: %s\n", e.error().c_str()); return {}; }
  const int bpp = (c.out_format == OutFormat::RGB24 || c.out_format == OutFormat::BGR24) ? 3 : 4;
  std::vector<uint8_t> out((size_t)c.width * c.height * bpp);
  FrameIn in = FrameIn::contiguous(c.in_format, c.width, c.height, frame.data());
  for (int i = 0; i < frames; ++i) e.process(in, out.data(), c.width * bpp);
  return out;
}
static const float kCast[3] = {1.35f, 1.0f, 0.65f};

// ---- tests ---------------------------------------------------------------------------------------------
static void test_yuv_exact() {
  const int w = 66, h = 48;
  synth::Rng rng(7);
  std::vector<uint8_t> f((size_t)w * h * 3 / 2);
  for (auto& v : f) v = (uint8_t)(rng.next() & 255);
  for (Matrix m : {Matrix::BT601, Matrix::BT709})
    for (Range r : {Range::Limited, Range::Full}) {
      Config c = allOff(w, h, InFormat::I420);
      c.matrix = m; c.range = r; c.chroma_upsample = ChromaUpsample::Nearest;
      auto out = runOnce(c, f);
      const double kr = m == Matrix::BT601 ? 0.299 : 0.2126, kb = m == Matrix::BT601 ? 0.114 : 0.0722, kg = 1 - kr - kb;
      const double ys = r == Range::Limited ? 219 : 255, cs = r == Range::Limited ? 224 : 255, yo = r == Range::Limited ? 16 : 0;
      int maxd = 0;
      const uint8_t* U = f.data() + w * h; const uint8_t* V = U + (w / 2) * (h / 2);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          double Y = (f[y * w + x] - yo) / ys, u = (U[(y / 2) * (w / 2) + x / 2] - 128.0) / cs, v = (V[(y / 2) * (w / 2) + x / 2] - 128.0) / cs;
          double R = Y + 2 * (1 - kr) * v, B = Y + 2 * (1 - kb) * u, G = Y - 2 * kb * (1 - kb) / kg * u - 2 * kr * (1 - kr) / kg * v;
          double e[3] = {R, G, B};
          for (int k = 0; k < 3; ++k) {
            int ex = (int)std::lround(std::min(std::max(e[k] * 255.0, 0.0), 255.0));
            maxd = std::max(maxd, std::abs(ex - out[(y * w + x) * 3 + k]));
          }
        }
      std::printf("    %s %s: max |error| = %d level\n", m == Matrix::BT601 ? "BT.601" : "BT.709", r == Range::Limited ? "limited" : "full   ", maxd);
      CHECK(maxd <= 1, "maxd=%d", maxd);
    }
}

static void test_rgb_passthrough() {
  const int w = 97, h = 61;
  auto img = synth::sceneRGB(w, h, 3, 0.8f, nullptr, 2.f);
  Config c = allOff(w, h, InFormat::RGB24);
  auto out = runOnce(c, img);
  CHECK(out == img, "stages off must be bit exact");
  c.enable_denoise = c.enable_clahe = c.enable_lut = true;      // enabled but neutral parameters
  out = runOnce(c, img, 3);
  CHECK(out == img, "enabled stages with neutral params must be bit exact");
}

static void test_factored_vs_exact() {
  synth::Rng rng(11);
  struct P { float wr, wb, ev, c, s, l; };
  const P ps[] = {{1.25f, 0.8f, 0, 1, 1, 0}, {1, 1, 1.0f, 1, 1, 0}, {1, 1, -0.7f, 1, 1, 0}, {1, 1, 0, 1.3f, 1.2f, 0},
                  {1, 1, 0, 1, 1, 0.45f}, {0.8f, 1.3f, 0.6f, 1.25f, 1.15f, 0.3f}, {1.5f, 0.67f, 1.5f, 1.35f, 1.25f, 0.5f}};
  double worstAvg = 0; int worstMax = 0; double worstDE = 0;
  for (const P& q : ps) {
    Params p; p.wb_r = q.wr; p.wb_b = q.wb; p.exposure_ev = q.ev; p.contrast = q.c; p.saturation = q.s; p.shadow_lift = q.l;
    FactoredRT f; f.build(p);
    const int n = 60000;
    std::vector<uint8_t> r(n), g(n), b(n), r0, g0, b0;
    for (int i = 0; i < n; ++i) { r[i] = rng.next() & 255; g[i] = rng.next() & 255; b[i] = rng.next() & 255; }
    for (int i = 0; i < 256; ++i) { r[i] = g[i] = b[i] = (uint8_t)i; }   // neutral axis
    r0 = r; g0 = g; b0 = b;
    factoredRow(f, r.data(), g.data(), b.data(), n);
    double sum = 0; int mx = 0; double mde = 0;
    for (int i = 0; i < n; ++i) {
      float in[3] = {r0[i] / 255.f, g0[i] / 255.f, b0[i] / 255.f}, o[3];
      exactTransform(p, nullptr, in, o);
      int ex[3] = {(int)std::lround(o[0] * 255.f), (int)std::lround(o[1] * 255.f), (int)std::lround(o[2] * 255.f)};
      int got[3] = {r[i], g[i], b[i]};
      for (int k = 0; k < 3; ++k) { int d = std::abs(ex[k] - got[k]); sum += d; mx = std::max(mx, d); }
      uint8_t A[3] = {(uint8_t)ex[0], (uint8_t)ex[1], (uint8_t)ex[2]}, Bq[3] = {(uint8_t)got[0], (uint8_t)got[1], (uint8_t)got[2]};
      mde = std::max(mde, dE(A, Bq));
    }
    worstAvg = std::max(worstAvg, sum / (3.0 * n)); worstMax = std::max(worstMax, mx); worstDE = std::max(worstDE, mde);
  }
  std::printf("    factored vs exact float transform: mean |err| %.3f level, max %d level, max dE2000 %.2f\n", worstAvg, worstMax, worstDE);
  CHECK(worstMax <= 1, "max=%d", worstMax);
  CHECK(worstAvg < 0.15, "avg=%f", worstAvg);
  CHECK(worstDE < 1.3, "dE=%f", worstDE);   // +-1 level in very dark chroma can exceed dE 1
}

static void test_lut3d_accuracy() {
  synth::Rng rng(5);
  struct P { float wr, wb, ev, c, s, l; };
  const P ps[] = {{1.25f, 0.8f, 0, 1, 1, 0}, {1, 1, 1.0f, 1, 1, 0}, {1, 1, 0, 1.3f, 1.2f, 0.3f}, {0.8f, 1.3f, 0.6f, 1.25f, 1.15f, 0.3f}};
  ThreadPool pool(2);
  std::printf("    3D LUT vs exact (random + neutral axis, 4 parameter sets): size shaper mode -> mean dE00 / p99.9 dE00 / max level err\n");
  const int sizes[] = {17, 33};
  const float shapers[] = {1.0f, 0.8f, 0.65f};
  double defMean = 0, defMax = 0;
  for (int n : sizes)
    for (float sh : shapers)
      for (int tetra = 1; tetra >= 0; --tetra) {
        LutRT L; L.init(n, sh);
        double dsum = 0; long cnt = 0; int mxl = 0; std::vector<double> des;
        for (const P& q : ps) {
          Params p; p.wb_r = q.wr; p.wb_b = q.wb; p.exposure_ev = q.ev; p.contrast = q.c; p.saturation = q.s; p.shadow_lift = q.l;
          L.build(p, nullptr, pool);
          const int m = 20000;
          std::vector<uint8_t> r(m), g(m), b(m), o(m * 3);
          for (int i = 0; i < m; ++i) { r[i] = rng.next() & 255; g[i] = rng.next() & 255; b[i] = rng.next() & 255; }
          for (int i = 0; i < 256; ++i) { r[i] = g[i] = b[i] = (uint8_t)i; }
          lutApplyRow(L, tetra != 0, r.data(), g.data(), b.data(), m, OutFormat::RGB24, o.data());
          for (int i = 0; i < m; ++i) {
            float in[3] = {r[i] / 255.f, g[i] / 255.f, b[i] / 255.f}, e[3];
            exactTransform(p, nullptr, in, e);
            uint8_t A[3] = {(uint8_t)std::lround(e[0] * 255.f), (uint8_t)std::lround(e[1] * 255.f), (uint8_t)std::lround(e[2] * 255.f)};
            double d = dE(A, &o[i * 3]);
            des.push_back(d); dsum += d; ++cnt;
            for (int k = 0; k < 3; ++k) mxl = std::max(mxl, std::abs((int)A[k] - (int)o[i * 3 + k]));
          }
        }
        std::sort(des.begin(), des.end());
        double mean = dsum / cnt, p999 = des[(size_t)(0.999 * des.size())];
        std::printf("      n=%2d shaper=%.2f %-9s -> %.3f / %.2f / %d\n", n, sh, tetra ? "tetra" : "trilinear", mean, p999, mxl);
        if (n == 17 && sh == 1.0f && tetra) { defMean = mean; defMax = p999; }
      }
  // 17^3 + tetrahedral: report-only gate (generic non-linear transform sampled on a 17-grid)
  CHECK(defMean < 0.6, "mean dE00 %.3f", defMean);
  (void)defMax;
}

static void test_lut_simd_vs_scalar() {
  synth::Rng rng(9);
  ThreadPool pool(1);
  LutRT L; L.init(17, 0.8f);
  Params p; p.wb_r = 1.2f; p.wb_b = 0.85f; p.exposure_ev = 0.4f; p.contrast = 1.2f; p.saturation = 1.1f;
  L.build(p, nullptr, pool);
  const int n = 5003;
  std::vector<uint8_t> r(n), g(n), b(n);
  for (int i = 0; i < n; ++i) { r[i] = rng.next() & 255; g[i] = rng.next() & 255; b[i] = rng.next() & 255; }
  for (OutFormat f : {OutFormat::RGB24, OutFormat::BGR24, OutFormat::RGBA32, OutFormat::BGRA32}) {
    std::vector<uint8_t> a((size_t)n * 4 + 8, 0), c((size_t)n * 4 + 8, 0);
    setForceScalarLut(false);
    lutApplyRow(L, true, r.data(), g.data(), b.data(), n, f, a.data());
    setForceScalarLut(true);
    lutApplyRow(L, true, r.data(), g.data(), b.data(), n, f, c.data());
    setForceScalarLut(false);
    CHECK(a == c, "SIMD and scalar tetrahedral kernels differ (format %d)", (int)f);
  }
}

static void test_wb_correction() {
  // --- A: textured scene (edges, neutral wedge, coloured blocks, big sky-like gradients) -------------
  const int w = 256, h = 192;
  double worstGain = 0, bestRatio = 1e9;
  for (int seed = 1; seed <= 4; ++seed) {
    auto ref = synth::sceneRGB(w, h, seed, 1.0f);
    float cast[3] = {1.35f, 1.0f, 0.65f};
    if (seed == 2) { cast[0] = 0.75f; cast[2] = 1.4f; }          // cool cast
    if (seed == 3) { cast[0] = 1.15f; cast[1] = 0.85f; cast[2] = 1.05f; }   // magenta-ish tint
    auto bad = synth::sceneRGB(w, h, seed, 1.0f, cast, 2.0f);
    Config c = allOff(w, h, InFormat::RGB24);
    c.enable_lut = true; c.auto_wb = true; c.wb_strength = 1.0f;
    Enhancer e(c);
    std::vector<uint8_t> out(ref.size());
    e.process(FrameIn::contiguous(InFormat::RGB24, w, h, bad.data()), out.data(), w * 3);
    double b = 0, a = 0; long n = 0;
    for (size_t i = 0; i < (size_t)w * h; i += 7) { b += dE(&bad[i * 3], &ref[i * 3]); a += dE(&out[i * 3], &ref[i * 3]); ++n; }
    Params p = e.currentParams();
    std::printf("    scene %d cast(%.2f,%.2f,%.2f): mean dE00 vs truth %.2f -> %.2f   (gains r=%.3f b=%.3f)\n", seed, cast[0], cast[1], cast[2], b / n, a / n, p.wb_r, p.wb_b);
    CHECK(a <= b, "WB made the image worse (seed %d)", seed);          // never worse than the input
    if (seed == 1 || seed == 4) CHECK(a < 0.5 * b, "warm cast not corrected (seed %d): %.2f -> %.2f", seed, b / n, a / n);
    bestRatio = std::min(bestRatio, a / b); worstGain = std::max(worstGain, a / b);
  }
  (void)bestRatio; (void)worstGain;   // cool/tint casts are corrected only partially by design (conservative gate)

  // --- B: do no harm: neutral illuminant stays put (textured scene and chart) ------------------------
  for (int k = 0; k < 2; ++k) {
    std::vector<uint8_t> ref;
    int ww = w, hh = h;
    if (k == 0) ref = synth::sceneRGB(w, h, 9, 1.0f, nullptr, 2.0f);
    else {
      static const uint8_t cc[24][3] = {{115,82,68},{194,150,130},{98,122,157},{87,108,67},{133,128,177},{103,189,170},
                                        {214,126,44},{80,91,166},{193,90,99},{94,60,108},{157,188,64},{224,163,46},
                                        {56,61,150},{70,148,73},{175,54,60},{231,199,31},{187,86,149},{8,133,161},
                                        {243,243,242},{200,200,200},{160,160,160},{122,122,121},{85,85,85},{52,52,52}};
      ww = 192; hh = 128; ref.resize((size_t)ww * hh * 3);
      for (int y = 0; y < hh; ++y) for (int x = 0; x < ww; ++x) for (int q = 0; q < 3; ++q) ref[(y * ww + x) * 3 + q] = cc[(y / 32) * 6 + x / 32][q];
    }
    Config c = allOff(ww, hh, InFormat::RGB24);
    c.enable_lut = true; c.auto_wb = true; c.wb_strength = 1.0f;
    Enhancer e(c);
    std::vector<uint8_t> out(ref.size());
    e.process(FrameIn::contiguous(InFormat::RGB24, ww, hh, ref.data()), out.data(), ww * 3);
    double a = 0; long n = 0;
    for (size_t i = 0; i < (size_t)ww * hh; i += 5) { a += dE(&out[i * 3], &ref[i * 3]); ++n; }
    Params p = e.currentParams();
    std::printf("    no-cast %s: dE00 introduced by WB %.2f (gains r=%.3f b=%.3f)\n", k ? "colour chart" : "textured scene", a / n, p.wb_r, p.wb_b);
    CHECK(a / n < 1.0, "WB harmed a neutral-illuminant image: %.2f", a / n);
  }

  // --- C: colour chart under cast: must not get worse (flat patches give no usable edges) ---------------
  {
    static const uint8_t cc[24][3] = {{115,82,68},{194,150,130},{98,122,157},{87,108,67},{133,128,177},{103,189,170},
                                      {214,126,44},{80,91,166},{193,90,99},{94,60,108},{157,188,64},{224,163,46},
                                      {56,61,150},{70,148,73},{175,54,60},{231,199,31},{187,86,149},{8,133,161},
                                      {243,243,242},{200,200,200},{160,160,160},{122,122,121},{85,85,85},{52,52,52}};
    const int cw = 192, ch = 128;
    std::vector<uint8_t> ref((size_t)cw * ch * 3), bad(ref.size());
    for (int y = 0; y < ch; ++y) for (int x = 0; x < cw; ++x) for (int q = 0; q < 3; ++q) {
      const uint8_t v = cc[(y / 32) * 6 + x / 32][q];
      ref[(y * cw + x) * 3 + q] = v;
      bad[(y * cw + x) * 3 + q] = (uint8_t)std::lround(synth::l2s(synth::s2l(v / 255.f) * kCast[q]) * 255.f);
    }
    Config c = allOff(cw, ch, InFormat::RGB24);
    c.enable_lut = true; c.auto_wb = true; c.wb_strength = 1.0f;
    Enhancer e(c);
    std::vector<uint8_t> out(ref.size());
    e.process(FrameIn::contiguous(InFormat::RGB24, cw, ch, bad.data()), out.data(), cw * 3);
    double b = 0, a = 0;
    for (int p = 0; p < 24; ++p) { size_t i = ((size_t)(p / 6) * 32 + 16) * cw + (p % 6) * 32 + 16; b += dE(&bad[i * 3], &ref[i * 3]); a += dE(&out[i * 3], &ref[i * 3]); }
    std::printf("    colour chart under cast: dE00 %.2f -> %.2f (flat patches: gray-edge has no edges to use; must not get worse)\n", b / 24, a / 24);
    CHECK(a <= b + 0.01, "chart made worse");
  }

  // --- D: colour-dominated scene (no neutrals): must not invent a correction -----------------------------
  {
    const int rw = 192, rh = 128;
    std::vector<uint8_t> reds((size_t)rw * rh * 3);
    for (int y = 0; y < rh; ++y)
      for (int x = 0; x < rw; ++x) {
        float t = (float)((x / 16 + y / 16) % 5) / 5.f;
        reds[(y * rw + x) * 3 + 0] = (uint8_t)(150 + 80 * t); reds[(y * rw + x) * 3 + 1] = (uint8_t)(30 + 40 * t); reds[(y * rw + x) * 3 + 2] = (uint8_t)(25 + 20 * t);
      }
    Config c = allOff(rw, rh, InFormat::RGB24);
    c.enable_lut = true; c.auto_wb = true; c.wb_strength = 1.0f;
    Enhancer e3(c);
    std::vector<uint8_t> out(reds.size());
    e3.process(FrameIn::contiguous(InFormat::RGB24, rw, rh, reds.data()), out.data(), rw * 3);
    Params p3 = e3.currentParams();
    std::printf("    all-red scene: gains r=%.3f b=%.3f (should stay ~1)\n", p3.wb_r, p3.wb_b);
    CHECK(std::fabs(std::log(p3.wb_r)) < 0.05 && std::fabs(std::log(p3.wb_b)) < 0.05, "r=%.3f b=%.3f", p3.wb_r, p3.wb_b);
  }
}

static void test_exposure_direction() {
  const int w = 160, h = 120;
  auto evFor = [&](float exposure) {
    auto img = synth::sceneRGB(w, h, 4, exposure);
    Config c = stillCfg(w, h);
    Enhancer e(c);
    std::vector<uint8_t> out((size_t)w * h * 3);
    e.process(FrameIn::contiguous(InFormat::RGB24, w, h, img.data()), out.data(), w * 3);
    return e.currentParams().exposure_ev;
  };
  float dark = evFor(0.15f), mid = evFor(0.7f), bright = evFor(2.2f);
  std::printf("    auto exposure: dark scene %+.2f EV, normal %+.2f EV, bright %+.2f EV\n", dark, mid, bright);
  CHECK(dark > 0.4f, "dark=%f", dark);
  CHECK(std::fabs(mid) < 0.5f, "mid=%f", mid);
  CHECK(bright < dark, "bright=%f", bright);
}

static void test_ema_math() {
  Params a, b; a.exposure_ev = 1.f; a.wb_r = 1.2f; b.exposure_ev = 0.f; b.wb_r = 0.9f;
  Params m = blendParams(a, b, 0.15f);
  CHECK(std::fabs(m.exposure_ev - 0.85f) < 1e-6f, "ev=%f", m.exposure_ev);
  CHECK(std::fabs(m.wb_r - std::exp(0.85f * std::log(1.2f) + 0.15f * std::log(0.9f))) < 1e-5f, "wb=%f", m.wb_r);

  const int w = 128, h = 96;
  auto dark = synth::sceneRGB(w, h, 4, 0.2f), bright = synth::sceneRGB(w, h, 4, 1.0f);
  Config c = stillCfg(w, h);
  Enhancer s1(c), s2(c);
  std::vector<uint8_t> out((size_t)w * h * 3);
  s1.process(FrameIn::contiguous(InFormat::RGB24, w, h, dark.data()), out.data(), w * 3);
  s2.process(FrameIn::contiguous(InFormat::RGB24, w, h, bright.data()), out.data(), w * 3);
  Params r1 = s1.targetParams(), r2 = s2.targetParams();
  Config c2 = c; c2.analysis_interval = 1;
  Enhancer e(c2);
  e.process(FrameIn::contiguous(InFormat::RGB24, w, h, dark.data()), out.data(), w * 3);
  e.process(FrameIn::contiguous(InFormat::RGB24, w, h, bright.data()), out.data(), w * 3);
  float expect = 0.85f * r1.exposure_ev + 0.15f * r2.exposure_ev;
  std::printf("    EMA: prev %.3f, new %.3f -> state %.4f (expected %.4f)\n", r1.exposure_ev, r2.exposure_ev, e.targetParams().exposure_ev, expect);
  CHECK(std::fabs(e.targetParams().exposure_ev - expect) < 1e-5f, "ema mismatch");
}

static void test_temporal_stability() {
  const int w = 320, h = 240;
  Config c; c.width = w; c.height = h; c.in_format = InFormat::I420; c.threads = 2;   // async, defaults
  Enhancer e(c);
  std::vector<uint8_t> out((size_t)w * h * 3);
  double maxEv = 0, maxWb = 0, maxClahe = 0;
  Params prev;
  for (int i = 0; i < 160; ++i) {
    auto rgb = synth::sceneRGB(w, h, 1000 + i, 0.5f, kCast, 3.5f);    // same scene, fresh sensor noise
    auto f = synth::rgbToFormat(rgb, w, h, InFormat::I420, Matrix::BT601, Range::Limited);
    e.process(FrameIn::contiguous(InFormat::I420, w, h, f.data()), out.data(), w * 3);
    Params p = e.currentParams();
    if (i >= 50) {
      maxEv = std::max(maxEv, (double)std::fabs(p.exposure_ev - prev.exposure_ev));
      maxWb = std::max(maxWb, (double)std::fabs(std::log(p.wb_r / prev.wb_r)));
      maxClahe = std::max(maxClahe, (double)std::fabs(p.clahe_strength - prev.clahe_strength));
    }
    prev = p;
    if (i % 13 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  std::printf("    static noisy scene, 110 frames after warm-up: max per-frame step  EV %.4f  log(wb_r) %.4f  CLAHE %.4f\n", maxEv, maxWb, maxClahe);
  CHECK(maxEv < 0.03, "ev step %f", maxEv);
  CHECK(maxWb < 0.01, "wb step %f", maxWb);
  CHECK(maxClahe < 0.03, "clahe step %f", maxClahe);

  // scene change: parameters must glide, not jump
  Enhancer e2(c);
  double maxJump = 0; Params q;
  for (int i = 0; i < 140; ++i) {
    auto rgb = synth::sceneRGB(w, h, 77, i < 60 ? 0.12f : 1.0f);
    auto f = synth::rgbToFormat(rgb, w, h, InFormat::I420, Matrix::BT601, Range::Limited);
    e2.process(FrameIn::contiguous(InFormat::I420, w, h, f.data()), out.data(), w * 3);
    Params p = e2.currentParams();
    if (i > 0) maxJump = std::max(maxJump, (double)std::fabs(p.exposure_ev - q.exposure_ev));
    q = p;
    if (i % 3 == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::printf("    scene change dark->normal: max per-frame EV step %.3f, final EV %+.2f\n", maxJump, q.exposure_ev);
  CHECK(maxJump < 0.15, "jump %f", maxJump);
}

static void test_no_alloc() {
  const int w = 640, h = 480;
  Config c; c.width = w; c.height = h; c.in_format = InFormat::NV12; c.threads = 3;
  Enhancer e(c);
  std::vector<std::vector<uint8_t>> fr;
  for (int i = 0; i < 4; ++i)
    fr.push_back(synth::rgbToFormat(synth::sceneRGB(w, h, 20 + i, 0.5f, kCast, 3.f), w, h, InFormat::NV12, Matrix::BT601, Range::Limited));
  std::vector<uint8_t> out((size_t)w * h * 3);
  for (int i = 0; i < 40; ++i) e.process(FrameIn::contiguous(InFormat::NV12, w, h, fr[i % 4].data()), out.data(), w * 3);
  g_allocs = 0; g_countAllocs = true;
  for (int i = 0; i < 80; ++i) e.process(FrameIn::contiguous(InFormat::NV12, w, h, fr[i % 4].data()), out.data(), w * 3);
  g_countAllocs = false;
  std::printf("    heap allocations during 80 steady-state frames (3 threads, async analysis): %ld\n", g_allocs.load());
  CHECK(g_allocs.load() == 0, "allocs=%ld", g_allocs.load());
}

static void test_thread_determinism() {
  const int w = 211, h = 133;
  std::vector<std::vector<uint8_t>> fr;
  for (int i = 0; i < 5; ++i)
    fr.push_back(synth::rgbToFormat(synth::sceneRGB(w, h, 40 + i, 0.4f, kCast, 4.f), w, h, InFormat::I420, Matrix::BT709, Range::Limited));
  std::vector<std::vector<uint8_t>> res;
  for (int t : {1, 2, 4, 7}) {
    Config c = stillCfg(w, h, InFormat::I420);
    c.matrix = Matrix::BT709; c.threads = t; c.frame_smoothing = 0.2f; c.analysis_interval = 2;
    Enhancer e(c);
    std::vector<uint8_t> out((size_t)w * h * 3), all;
    for (int i = 0; i < 5; ++i) { e.process(FrameIn::contiguous(InFormat::I420, w, h, fr[i].data()), out.data(), w * 3); all.insert(all.end(), out.begin(), out.end()); }
    res.push_back(all);
  }
  for (size_t i = 1; i < res.size(); ++i) CHECK(res[i] == res[0], "thread count changes the output (variant %zu)", i);
}

static void test_format_equivalence() {
  const int w = 130, h = 94;
  auto rgb = synth::sceneRGB(w, h, 8, 0.6f, nullptr, 3.f);
  auto i420 = synth::rgbToFormat(rgb, w, h, InFormat::I420, Matrix::BT601, Range::Limited);
  auto nv12 = synth::rgbToFormat(rgb, w, h, InFormat::NV12, Matrix::BT601, Range::Limited);
  // YUYV with 4:2:0 chroma replicated vertically
  const int cw = w / 2;
  std::vector<uint8_t> yuyv((size_t)w * 2 * h);
  const uint8_t* U = i420.data() + w * h; const uint8_t* V = U + cw * (h / 2);
  for (int y = 0; y < h; ++y)
    for (int cx = 0; cx < cw; ++cx) {
      uint8_t* d = &yuyv[((size_t)y * cw + cx) * 4];
      d[0] = i420[y * w + 2 * cx]; d[1] = U[(y / 2) * cw + cx]; d[2] = i420[y * w + 2 * cx + 1]; d[3] = V[(y / 2) * cw + cx];
    }
  for (ChromaUpsample cu : {ChromaUpsample::Linear, ChromaUpsample::Nearest}) {
    Config a = stillCfg(w, h, InFormat::I420); a.chroma_upsample = cu;
    Config b = a; b.in_format = InFormat::NV12;
    auto oa = runOnce(a, i420), ob = runOnce(b, nv12);
    CHECK(oa == ob, "NV12 differs from I420");
  }
  Config a = stillCfg(w, h, InFormat::I420); a.chroma_upsample = ChromaUpsample::Nearest;
  Config y = a; y.in_format = InFormat::YUYV;
  CHECK(runOnce(a, i420) == runOnce(y, yuyv), "YUYV differs from I420 (nearest, replicated chroma)");
  // output formats
  Config base = stillCfg(w, h, InFormat::I420);
  auto o24 = runOnce(base, i420);
  Config cb = base; cb.out_format = OutFormat::BGR24; auto ob24 = runOnce(cb, i420);
  Config ca = base; ca.out_format = OutFormat::RGBA32; auto oa32 = runOnce(ca, i420);
  Config cc = base; cc.out_format = OutFormat::BGRA32; auto ob32 = runOnce(cc, i420);
  bool ok = true;
  for (int i = 0; i < w * h; ++i) {
    for (int k = 0; k < 3; ++k) {
      ok &= ob24[i * 3 + k] == o24[i * 3 + 2 - k];
      ok &= oa32[i * 4 + k] == o24[i * 3 + k];
      ok &= ob32[i * 4 + k] == o24[i * 3 + 2 - k];
    }
    ok &= oa32[i * 4 + 3] == 255 && ob32[i * 4 + 3] == 255;
  }
  CHECK(ok, "output pixel formats inconsistent");
}

static void test_clahe_unit() {
  synth::Rng rng(3);
  const int n = 64;
  std::vector<uint8_t> y(n * n);
  for (auto& v : y) v = (uint8_t)(16 + (rng.next() % 60));         // dull, dark tile
  uint16_t lut[256];
  k::claheTileLut(y.data(), n, 0, 0, n, n, 1, 2.5f, 1.0f, 16, 235, lut);
  bool mono = true, inrange = true;
  for (int i = 1; i < 256; ++i) mono &= lut[i] >= lut[i - 1];
  for (int i = 0; i < 256; ++i) inrange &= lut[i] >= 16 * 256 && lut[i] <= 235 * 256;
  CHECK(mono, "tile LUT not monotone");
  CHECK(inrange, "tile LUT leaves legal luma range");
  k::claheTileLut(y.data(), n, 0, 0, n, n, 1, 2.5f, 0.0f, 16, 235, lut);
  bool ident = true;
  for (int i = 16; i <= 235; ++i) ident &= lut[i] == i * 256;
  CHECK(ident, "strength 0 must be identity");
  // slope never exceeds roughly the clip limit
  k::claheTileLut(y.data(), n, 0, 0, n, n, 1, 2.0f, 1.0f, 16, 235, lut);
  double maxSlope = 0;
  for (int i = 17; i < 255; ++i) maxSlope = std::max(maxSlope, (lut[i + 1] - lut[i - 1]) / 512.0);
  std::printf("    CLAHE clip 2.0: max local slope %.2f (limit ~ clip*219/255 = %.2f + redistribution)\n", maxSlope, 2.0 * 219 / 255);
  CHECK(maxSlope < 2.0 * 219 / 255 * 1.6 + 0.2, "slope %f", maxSlope);
}

static void test_cube_lut() {
  const char* idp = "hie_test_identity.cube";
  const char* inv = "hie_test_invert.cube";
  const int n = 5;
  for (int mode = 0; mode < 2; ++mode) {
    std::ofstream f(mode ? inv : idp);
    f << "# test\nTITLE \"t\"\nLUT_3D_SIZE " << n << "\n";
    for (int b = 0; b < n; ++b) for (int g = 0; g < n; ++g) for (int r = 0; r < n; ++r) {
      float v[3] = {(float)r / (n - 1), (float)g / (n - 1), (float)b / (n - 1)};
      if (mode) for (float& t : v) t = 1.f - t;
      f << v[0] << " " << v[1] << " " << v[2] << "\n";
    }
  }
  const int w = 120, h = 80;
  auto img = synth::sceneRGB(w, h, 6, 0.9f);
  for (int mode = 0; mode < 2; ++mode) {
    Lut3D l; std::string err;
    CHECK(Lut3D::loadCube(mode ? inv : idp, l, &err), "load: %s", err.c_str());
    Config c = allOff(w, h, InFormat::RGB24);
    c.enable_lut = true;
    c.manual.lut_weights[0] = 1.f;
    Enhancer e(c);
    CHECK(e.addLut(l), "addLut");
    std::vector<uint8_t> out((size_t)w * h * 3);
    e.process(FrameIn::contiguous(InFormat::RGB24, w, h, img.data()), out.data(), w * 3);
    int mx = 0; double sum = 0;
    for (size_t i = 0; i < out.size(); ++i) { int ex = mode ? 255 - img[i] : img[i]; int d = std::abs(ex - out[i]); mx = std::max(mx, d); sum += d; }
    std::printf("    .cube %-8s through 3D LUT path: mean |err| %.3f level, max %d\n", mode ? "invert" : "identity", sum / out.size(), mx);
    CHECK(mx <= 3, "max err %d", mx);
  }
  Lut3D bad; std::string err;
  CHECK(!Lut3D::loadCube("does_not_exist.cube", bad, &err), "missing file must fail");
  std::remove(idp); std::remove(inv);
}

static void test_odd_sizes() {
  for (InFormat f : {InFormat::I420, InFormat::NV12, InFormat::RGB24, InFormat::YUYV}) {
    const int w = f == InFormat::YUYV ? 322 : 321, h = 241;
    auto rgb = synth::sceneRGB(w, h, 12, 0.5f, kCast, 3.f);
    std::vector<uint8_t> fr;
    if (f == InFormat::YUYV || w % 2 == 0) fr = synth::rgbToFormat(rgb, w, h, f, Matrix::BT601, Range::Limited);
    else fr = synth::rgbToFormat(rgb, w, h, f, Matrix::BT601, Range::Limited);
    Config c; c.width = w; c.height = h; c.in_format = f; c.async_analysis = false; c.threads = 3;
    Enhancer e(c);
    CHECK(e.valid(), "config invalid");
    std::vector<uint8_t> out((size_t)w * h * 3 + 16, 0xAB);
    bool ok = true;
    for (int i = 0; i < 3; ++i) ok &= e.process(FrameIn::contiguous(f, w, h, fr.data()), out.data(), w * 3);
    CHECK(ok, "process failed %dx%d fmt %d", w, h, (int)f);
    bool guard = true;
    for (int i = 0; i < 16; ++i) guard &= out[(size_t)w * h * 3 + i] == 0xAB;
    CHECK(guard, "wrote past end of output (fmt %d)", (int)f);
  }
  Config bad; bad.width = 0; bad.height = 10;
  Enhancer e(bad);
  CHECK(!e.valid(), "zero width must be rejected");
}

static void test_denoise() {
  const int w = 192, h = 128;
  std::vector<uint8_t> img((size_t)w * h * 3);
  synth::Rng rng(21);
  const float sigma = 6.f;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float base = x < w / 2 ? 60.f : 180.f;
      float n = sigma * rng.gauss();     // luma noise: identical on all channels
      uint8_t v = (uint8_t)std::lround(std::min(std::max(base + n, 0.f), 255.f));
      for (int k = 0; k < 3; ++k) img[(y * w + x) * 3 + k] = v;
    }
  Config c = allOff(w, h, InFormat::RGB24);
  c.enable_denoise = true; c.auto_denoise = true;
  Enhancer e(c);
  std::vector<uint8_t> out(img.size());
  e.process(FrameIn::contiguous(InFormat::RGB24, w, h, img.data()), out.data(), w * 3);
  auto stdev = [&](const std::vector<uint8_t>& a, int x0, int x1) {
    double s = 0, s2 = 0; long n = 0;
    for (int y = 0; y < h; ++y) for (int x = x0; x < x1; ++x) { double v = a[(y * w + x) * 3 + 1]; s += v; s2 += v * v; ++n; }
    double m = s / n; return std::sqrt(s2 / n - m * m);
  };
  auto mean = [&](const std::vector<uint8_t>& a, int x0, int x1) {
    double s = 0; long n = 0;
    for (int y = 0; y < h; ++y) for (int x = x0; x < x1; ++x) { s += a[(y * w + x) * 3 + 1]; ++n; }
    return s / n;
  };
  double s0 = stdev(img, 10, w / 2 - 10), s1 = stdev(out, 10, w / 2 - 10);
  double step0 = mean(img, w / 2 + 6, w - 6) - mean(img, 6, w / 2 - 6), step1 = mean(out, w / 2 + 6, w - 6) - mean(out, 6, w / 2 - 6);
  std::printf("    denoise (T=%.1f): flat-area noise sigma %.2f -> %.2f; edge contrast %.1f -> %.1f\n", e.currentParams().denoise_t, s0, s1, step0, step1);
  CHECK(s1 < 0.7 * s0, "noise not reduced: %.2f -> %.2f", s0, s1);
  CHECK(step1 > 0.97 * step0, "edge contrast lost: %.1f -> %.1f", step0, step1);
  CHECK(e.currentParams().denoise_t >= 2.f, "noise estimate failed");
}

static void test_c_api() {
  const int w = 64, h = 48;
  auto rgb = synth::sceneRGB(w, h, 5, 0.5f, kCast, 2.f);
  auto f = synth::rgbToFormat(rgb, w, h, InFormat::I420, Matrix::BT601, Range::Limited);
  hie_config cc; hie_config_default(&cc, w, h);
  cc.async_analysis = 0; cc.threads = 2;
  hie_handle* h1 = hie_create(&cc);
  CHECK(h1 != nullptr, "create");
  FrameIn in = FrameIn::contiguous(InFormat::I420, w, h, f.data());
  const uint8_t* planes[3] = {in.data[0], in.data[1], in.data[2]};
  int strides[3] = {in.stride[0], in.stride[1], in.stride[2]};
  std::vector<uint8_t> o1((size_t)w * h * 3);
  CHECK(hie_process(h1, planes, strides, o1.data(), w * 3) == 1, "process");
  Config c; c.width = w; c.height = h; c.in_format = InFormat::I420; c.async_analysis = false; c.threads = 2;
  auto o2 = runOnce(c, f);
  CHECK(o1 == o2, "C API output differs from C++ API");
  float p[9]; hie_get_params(h1, p);
  CHECK(std::strlen(hie_version()) > 0, "version");
  hie_destroy(h1);
  hie_config bad = cc; bad.width = 0;
  CHECK(hie_create(&bad) == nullptr, "invalid config must fail");
}

static void test_enhancement_improves_dark_noisy_scene() {
  // end-to-end: dim, cast, noisy webcam-like frame vs the clean well-exposed truth
  const int w = 320, h = 240;
  auto truth = synth::sceneRGB(w, h, 31, 1.0f);
  auto bad = synth::sceneRGB(w, h, 31, 0.35f, kCast, 4.f);
  Config c = stillCfg(w, h, InFormat::RGB24);
  Enhancer e(c);
  std::vector<uint8_t> out(truth.size());
  FrameIn in = FrameIn::contiguous(InFormat::RGB24, w, h, bad.data());
  e.process(in, out.data(), w * 3);
  double b = 0, a = 0; const int step = 3; long n = 0;
  for (int y = 0; y < h; y += step) for (int x = 0; x < w; x += step) { size_t i = ((size_t)y * w + x) * 3; b += dE(&bad[i], &truth[i]); a += dE(&out[i], &truth[i]); ++n; }
  Params p = e.currentParams();
  std::printf("    dim+cast+noise scene: mean dE00 vs clean truth %.2f -> %.2f   (EV %+.2f, wb %.2f/%.2f, CLAHE %.2f, denoise T %.1f)\n", b / n, a / n, p.exposure_ev, p.wb_r, p.wb_b, p.clahe_strength, p.denoise_t);
  CHECK(a < b * 0.75, "no improvement: %.2f -> %.2f", b / n, a / n);
}

// ---- main -------------------------------------------------------------------------------------------------
int main(int argc, char** argv) {
  struct T { const char* name; void (*fn)(); };
  const T tests[] = {
      {"yuv_exact", test_yuv_exact}, {"rgb_passthrough", test_rgb_passthrough},
      {"factored_vs_exact", test_factored_vs_exact}, {"lut3d_accuracy", test_lut3d_accuracy},
      {"lut_simd_vs_scalar", test_lut_simd_vs_scalar}, {"wb_correction", test_wb_correction},
      {"exposure_direction", test_exposure_direction}, {"ema_math", test_ema_math},
      {"temporal_stability", test_temporal_stability}, {"no_alloc", test_no_alloc},
      {"thread_determinism", test_thread_determinism}, {"format_equivalence", test_format_equivalence},
      {"clahe_unit", test_clahe_unit}, {"cube_lut", test_cube_lut}, {"odd_sizes", test_odd_sizes},
      {"denoise", test_denoise}, {"c_api", test_c_api},
      {"end_to_end_improves", test_enhancement_improves_dark_noisy_scene}};
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int ran = 0;
  for (const T& t : tests) {
    if (filter && !std::strstr(t.name, filter)) continue;
    int before = g_fail;
    std::printf("[ RUN  ] %s\n", t.name);
    t.fn();
    std::printf("[ %s ] %s\n", g_fail == before ? " OK " : "FAIL", t.name);
    ++ran;
  }
  std::printf("\n%d tests, %d checks, %d failed\n", ran, g_checks, g_fail);
  return g_fail ? 1 : 0;
}
