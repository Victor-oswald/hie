#include "lut3d.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define HIE_SSE41_PATH 1
#endif

namespace hie {

namespace {
std::atomic<int> g_forceScalar{0};
}
void setForceScalarLut(bool on) { g_forceScalar.store(on ? 1 : 0); }

// ---------------------------------------------------------------------------------------------
// Params helpers
// ---------------------------------------------------------------------------------------------
Params blendParams(const Params& a, const Params& b, float w) {
  auto L = [&](float x, float y) { return x + (y - x) * w; };
  Params r;
  r.wb_r = std::exp(L(std::log(a.wb_r), std::log(b.wb_r)));
  r.wb_b = std::exp(L(std::log(a.wb_b), std::log(b.wb_b)));
  r.exposure_ev = L(a.exposure_ev, b.exposure_ev);
  r.contrast = L(a.contrast, b.contrast);
  r.saturation = L(a.saturation, b.saturation);
  r.shadow_lift = L(a.shadow_lift, b.shadow_lift);
  r.clahe_strength = L(a.clahe_strength, b.clahe_strength);
  r.clahe_clip = L(a.clahe_clip, b.clahe_clip);
  r.denoise_t = L(a.denoise_t, b.denoise_t);
  for (int i = 0; i < kMaxLuts; ++i) r.lut_weights[i] = L(a.lut_weights[i], b.lut_weights[i]);
  return r;
}

bool lutIsNeutral(const Params& p) {
  const float e = 1e-4f;
  if (std::fabs(p.wb_r - 1.f) > e || std::fabs(p.wb_b - 1.f) > e) return false;
  if (std::fabs(p.exposure_ev) > e || std::fabs(p.contrast - 1.f) > e) return false;
  if (std::fabs(p.saturation - 1.f) > e || std::fabs(p.shadow_lift) > e) return false;
  for (int i = 0; i < kMaxLuts; ++i)
    if (std::fabs(p.lut_weights[i]) > e) return false;
  return true;
}

// ---------------------------------------------------------------------------------------------
// External LUTs
// ---------------------------------------------------------------------------------------------
Lut3D Lut3D::identity(int n) {
  Lut3D l;
  l.size = n;
  l.data.resize((size_t)n * n * n * 3);
  size_t i = 0;
  for (int b = 0; b < n; ++b)
    for (int g = 0; g < n; ++g)
      for (int r = 0; r < n; ++r) {
        l.data[i++] = (float)r / (n - 1);
        l.data[i++] = (float)g / (n - 1);
        l.data[i++] = (float)b / (n - 1);
      }
  return l;
}

bool Lut3D::loadCube(const std::string& path, Lut3D& out, std::string* err) {
  std::ifstream f(path);
  auto fail = [&](const char* m) {
    if (err) *err = m;
    return false;
  };
  if (!f) return fail("cannot open file");
  int n = 0;
  float dmin[3] = {0, 0, 0}, dmax[3] = {1, 1, 1};
  std::vector<float> d;
  std::string line;
  while (std::getline(f, line)) {
    size_t h = line.find('#');
    if (h != std::string::npos) line.resize(h);
    std::istringstream ss(line);
    std::string tok;
    if (!(ss >> tok)) continue;
    if (tok == "LUT_3D_SIZE") {
      ss >> n;
      if (n < 2 || n > 129) return fail("bad LUT_3D_SIZE");
      d.reserve((size_t)n * n * n * 3);
    } else if (tok == "DOMAIN_MIN") {
      ss >> dmin[0] >> dmin[1] >> dmin[2];
    } else if (tok == "DOMAIN_MAX") {
      ss >> dmax[0] >> dmax[1] >> dmax[2];
    } else if (tok == "LUT_1D_SIZE") {
      return fail("1D LUTs are not supported");
    } else if (tok == "TITLE" || tok == "LUT_3D_INPUT_RANGE" || tok == "LUT_1D_INPUT_RANGE") {
      continue;
    } else {
      char* e1;
      float v0 = std::strtof(tok.c_str(), &e1);
      if (*e1) return fail("unexpected token");
      float v1, v2;
      if (!(ss >> v1 >> v2)) return fail("short data line");
      d.push_back(v0);
      d.push_back(v1);
      d.push_back(v2);
    }
  }
  if (n == 0) return fail("missing LUT_3D_SIZE");
  if (d.size() != (size_t)n * n * n * 3) return fail("wrong number of data lines");
  for (size_t i = 0; i < d.size(); ++i) {
    int c = (int)(i % 3);
    float range = dmax[c] - dmin[c];
    if (range != 1.0f || dmin[c] != 0.0f) d[i] = (d[i] - dmin[c]) / (range > 0 ? range : 1.0f);
  }
  out.size = n;
  out.data = std::move(d);
  return true;
}

void sampleLut3D(const Lut3D& l, const float in[3], float out[3]) {
  const int n = l.size;
  int i0[3], i1[3];
  float f[3];
  for (int c = 0; c < 3; ++c) {
    float v = std::min(std::max(in[c], 0.f), 1.f) * (n - 1);
    int i = std::min((int)v, n - 2);
    i0[c] = i;
    i1[c] = i + 1;
    f[c] = v - i;
  }
  auto at = [&](int r, int g, int b, int c) { return l.data[(((size_t)b * n + g) * n + r) * 3 + c]; };
  for (int c = 0; c < 3; ++c) {
    float c00 = at(i0[0], i0[1], i0[2], c) * (1 - f[0]) + at(i1[0], i0[1], i0[2], c) * f[0];
    float c10 = at(i0[0], i1[1], i0[2], c) * (1 - f[0]) + at(i1[0], i1[1], i0[2], c) * f[0];
    float c01 = at(i0[0], i0[1], i1[2], c) * (1 - f[0]) + at(i1[0], i0[1], i1[2], c) * f[0];
    float c11 = at(i0[0], i1[1], i1[2], c) * (1 - f[0]) + at(i1[0], i1[1], i1[2], c) * f[0];
    float c0 = c00 * (1 - f[1]) + c10 * f[1];
    float c1 = c01 * (1 - f[1]) + c11 * f[1];
    out[c] = c0 * (1 - f[2]) + c1 * f[2];
  }
}

// ---------------------------------------------------------------------------------------------
// The colour transform
// ---------------------------------------------------------------------------------------------
static inline float toneCurve(float y, float lift, float c) {
  // shadow lift keeps black (y=0) and white (y=1) fixed, raises the toe.
  float y1 = y + lift * 2.5f * y * (1.f - y) * (1.f - y);
  // S-curve with slope c at mid grey and fixed end points (monotone for c <= 1.5).
  float u = 2.f * y1 - 1.f;
  return y1 + (c - 1.f) * 0.5f * u * (1.f - u * u);
}

float toneCurveShared(float y, float lift, float c) { return toneCurve(y, lift, c); }

void exactTransform(const Params& p, const std::vector<Lut3D>* bank, const float in[3],
                    float out[3]) {
  float lin[3] = {srgbToLin(in[0]), srgbToLin(in[1]), srgbToLin(in[2])};
  // luminance-normalised gains so neutral luminance is unchanged by white balance
  float gr = p.wb_r, gg = 1.f, gb = p.wb_b;
  float norm = 0.2126f * gr + 0.7152f * gg + 0.0722f * gb;
  float ev = std::exp2(p.exposure_ev) / norm;
  lin[0] = std::min(lin[0] * gr * ev, 1.f);
  lin[1] = std::min(lin[1] * gg * ev, 1.f);
  lin[2] = std::min(lin[2] * gb * ev, 1.f);
  float d[3] = {linToSrgb(lin[0]), linToSrgb(lin[1]), linToSrgb(lin[2])};

  float yd = 0.2126f * d[0] + 0.7152f * d[1] + 0.0722f * d[2];
  float yt = toneCurve(yd, p.shadow_lift, p.contrast);
  if (yt != yd) {
    if (yd > 1e-4f) {
      float s = yt / yd;
      d[0] *= s; d[1] *= s; d[2] *= s;
    } else {
      float dy = yt - yd;
      d[0] += dy; d[1] += dy; d[2] += dy;
    }
  }
  if (p.saturation != 1.f) {
    float yy = 0.2126f * d[0] + 0.7152f * d[1] + 0.0722f * d[2];
    for (int c = 0; c < 3; ++c) d[c] = yy + p.saturation * (d[c] - yy);
  }
  for (int c = 0; c < 3; ++c) d[c] = std::min(std::max(d[c], 0.f), 1.f);

  if (bank) {
    float wsum = 0;
    for (size_t k = 0; k < bank->size() && k < (size_t)kMaxLuts; ++k) wsum += p.lut_weights[k];
    if (wsum > 1.f) wsum = 1.f / wsum; else wsum = 1.f;  // normalisation factor when over-unity
    float tot = 0;
    float acc[3] = {0, 0, 0};
    for (size_t k = 0; k < bank->size() && k < (size_t)kMaxLuts; ++k) {
      float w = p.lut_weights[k] * wsum;
      if (w <= 0) continue;
      float s[3];
      sampleLut3D((*bank)[k], d, s);
      for (int c = 0; c < 3; ++c) acc[c] += w * s[c];
      tot += w;
    }
    for (int c = 0; c < 3; ++c) d[c] = (1.f - tot) * d[c] + acc[c];
  }
  for (int c = 0; c < 3; ++c) out[c] = std::min(std::max(d[c], 0.f), 1.f);
}

// ---------------------------------------------------------------------------------------------
// LutRT
// ---------------------------------------------------------------------------------------------
void LutRT::init(int size, float shaper_gamma) {
  n = size;
  shaper = shaper_gamma;
  nodes.assign((size_t)n * n * n * 4, 0);
  sR = (uint32_t)n * n * 4;
  sG = (uint32_t)n * 4;
  sB = 4;
  axis.assign(n, 0.f);
  offR.resize(256);
  offG.resize(256);
  offB.resize(256);
  for (int v = 0; v < 256; ++v) {
    double pos = (n - 1) * std::pow(v / 255.0, (double)shaper);
    int i = (int)pos;
    if (i > n - 2) i = n - 2;
    double fr = pos - i;
    idx[v] = (uint8_t)i;
    frac[v] = (uint16_t)std::lround(fr * 4096.0);
    offR[v] = (uint32_t)i * sR;
    offG[v] = (uint32_t)i * sG;
    offB[v] = (uint32_t)i * sB;
  }
  identity = true;
}

void LutRT::build(const Params& p, const std::vector<Lut3D>* bank, ThreadPool& pool) {
  identity = lutIsNeutral(p);
  if (identity) return;
  for (int i = 0; i < n; ++i)
    axis[i] = (float)std::pow((double)i / (n - 1), 1.0 / (double)shaper);
  uint16_t* nd = nodes.data();
  const int N = n;
  const float* ax = axis.data();
  pool.run(N, [&](int r, int) {
    for (int g = 0; g < N; ++g)
      for (int b = 0; b < N; ++b) {
        float in[3] = {ax[r], ax[g], ax[b]}, o[3];
        exactTransform(p, bank, in, o);
        uint16_t* q = nd + (((size_t)r * N + g) * N + b) * 4;
        q[0] = (uint16_t)std::lround(o[0] * 65280.f);
        q[1] = (uint16_t)std::lround(o[1] * 65280.f);
        q[2] = (uint16_t)std::lround(o[2] * 65280.f);
        q[3] = 0;
      }
  });
}

// ---------------------------------------------------------------------------------------------
// Row appliers
// ---------------------------------------------------------------------------------------------
HIE_MV void packRow(const uint8_t* HIE_RESTRICT r, const uint8_t* HIE_RESTRICT g,
                    const uint8_t* HIE_RESTRICT b, int w, OutFormat fmt, uint8_t* HIE_RESTRICT d) {
  switch (fmt) {
    case OutFormat::RGB24:
      for (int x = 0; x < w; ++x) { d[3 * x] = r[x]; d[3 * x + 1] = g[x]; d[3 * x + 2] = b[x]; }
      break;
    case OutFormat::BGR24:
      for (int x = 0; x < w; ++x) { d[3 * x] = b[x]; d[3 * x + 1] = g[x]; d[3 * x + 2] = r[x]; }
      break;
    case OutFormat::RGBA32:
      for (int x = 0; x < w; ++x) { d[4 * x] = r[x]; d[4 * x + 1] = g[x]; d[4 * x + 2] = b[x]; d[4 * x + 3] = 255; }
      break;
    case OutFormat::BGRA32:
      for (int x = 0; x < w; ++x) { d[4 * x] = b[x]; d[4 * x + 1] = g[x]; d[4 * x + 2] = r[x]; d[4 * x + 3] = 255; }
      break;
  }
}

namespace {

template <OutFormat F>
HIE_AI void store(uint8_t* d, int x, int r, int g, int b) {
  if (F == OutFormat::RGB24) { d[3 * x] = (uint8_t)r; d[3 * x + 1] = (uint8_t)g; d[3 * x + 2] = (uint8_t)b; }
  else if (F == OutFormat::BGR24) { d[3 * x] = (uint8_t)b; d[3 * x + 1] = (uint8_t)g; d[3 * x + 2] = (uint8_t)r; }
  else if (F == OutFormat::RGBA32) { d[4 * x] = (uint8_t)r; d[4 * x + 1] = (uint8_t)g; d[4 * x + 2] = (uint8_t)b; d[4 * x + 3] = 255; }
  else { d[4 * x] = (uint8_t)b; d[4 * x + 1] = (uint8_t)g; d[4 * x + 2] = (uint8_t)r; d[4 * x + 3] = 255; }
}

// Tetrahedral interpolation, 12-bit fractions, 16-bit (x65280) node values.
// Vertex ordering is resolved without data-dependent branches (selects only): with fractions
// sorted wa >= wb >= wc the result is  v0 + wa*(v1-v0) + wb*(v2-v1) + wc*(v3-v2)  where
// v1 = v0 + e_max, v2 = v0 + (e_all - e_min), v3 = v0 + e_all.
struct TetraSel { int o1, o2, wa, wb, wc; };

HIE_AI TetraSel tetraSelect(int fr, int fg, int fb, int sR, int sG, int sB, int S) {
  const bool m1 = fr >= fg;
  const int maxrg = m1 ? fr : fg, minrg = m1 ? fg : fr;
  const int e1 = m1 ? sR : sG, e1b = m1 ? sG : sR;
  const bool m2 = maxrg >= fb, m4 = minrg <= fb;
  TetraSel t;
  t.wa = m2 ? maxrg : fb;
  t.wc = m4 ? minrg : fb;
  t.wb = fr + fg + fb - t.wa - t.wc;
  t.o1 = m2 ? e1 : sB;
  t.o2 = S - (m4 ? e1b : sB);
  return t;
}

template <OutFormat F>
HIE_AI void storePix(uint8_t* d, int x, uint32_t v) {   // v = R | G<<8 | B<<16
  if (F == OutFormat::RGB24) { d[3 * x] = (uint8_t)v; d[3 * x + 1] = (uint8_t)(v >> 8); d[3 * x + 2] = (uint8_t)(v >> 16); }
  else if (F == OutFormat::BGR24) { d[3 * x] = (uint8_t)(v >> 16); d[3 * x + 1] = (uint8_t)(v >> 8); d[3 * x + 2] = (uint8_t)v; }
  else if (F == OutFormat::RGBA32) { uint32_t o = v | 0xFF000000u; std::memcpy(d + 4 * x, &o, 4); }
  else { uint32_t o = ((v & 0xFFu) << 16) | (v & 0xFF00u) | ((v >> 16) & 0xFFu) | 0xFF000000u; std::memcpy(d + 4 * x, &o, 4); }
}

template <OutFormat F>
void tetraRowScalar(const LutRT& L, const uint8_t* r, const uint8_t* g, const uint8_t* b, int w,
                    uint8_t* d) {
  const uint16_t* nd = L.nodes.data();
  const uint32_t* oR = L.offR.data();
  const uint32_t* oG = L.offG.data();
  const uint32_t* oB = L.offB.data();
  const uint16_t* ft = L.frac;
  const int sR = (int)L.sR, sG = (int)L.sG, sB = (int)L.sB, S = sR + sG + sB;
  for (int x = 0; x < w; ++x) {
    const int R = r[x], G = g[x], B = b[x];
    const uint16_t* p0 = nd + oR[R] + oG[G] + oB[B];
    const TetraSel t = tetraSelect(ft[R], ft[G], ft[B], sR, sG, sB, S);
    const uint16_t* p1 = p0 + t.o1;
    const uint16_t* p2 = p0 + t.o2;
    const uint16_t* p3 = p0 + S;
    uint32_t v = 0;
    for (int c = 0; c < 3; ++c) {
      int v0 = p0[c], v1 = p1[c], v2 = p2[c], v3 = p3[c];
      int acc = (v0 << 12) + t.wa * (v1 - v0) + t.wb * (v2 - v1) + t.wc * (v3 - v2);
      v |= (uint32_t)((acc + (1 << 19)) >> 20) << (8 * c);
    }
    storePix<F>(d, x, v);
  }
}

#ifdef HIE_SSE41_PATH
// Same arithmetic as the scalar kernel, 4 lanes = (R,G,B,pad) of one pixel in SSE registers.
template <OutFormat F>
__attribute__((target("sse4.1"))) void tetraRowSse41(const LutRT& L, const uint8_t* r,
                                                     const uint8_t* g, const uint8_t* b, int w,
                                                     uint8_t* d) {
  const uint16_t* nd = L.nodes.data();
  const uint32_t* oR = L.offR.data();
  const uint32_t* oG = L.offG.data();
  const uint32_t* oB = L.offB.data();
  const uint16_t* ft = L.frac;
  const int sR = (int)L.sR, sG = (int)L.sG, sB = (int)L.sB, S = sR + sG + sB;
  const __m128i rnd = _mm_set1_epi32(1 << 19);
  for (int x = 0; x < w; ++x) {
    const int R = r[x], G = g[x], B = b[x];
    const uint16_t* p0 = nd + oR[R] + oG[G] + oB[B];
    const int fr = ft[R], fg = ft[G], fb = ft[B];
    const bool m1 = fr >= fg;
    const int maxrg = m1 ? fr : fg, minrg = m1 ? fg : fr;
    const int e1 = m1 ? sR : sG, e1b = m1 ? sG : sR;
    const bool m2 = maxrg >= fb, m4 = minrg <= fb;
    const int wa = m2 ? maxrg : fb, wc = m4 ? minrg : fb;
    const int wb = fr + fg + fb - wa - wc;
    const int o1 = m2 ? e1 : sB, o2 = S - (m4 ? e1b : sB);
    const __m128i v0 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)p0));
    const __m128i v1 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)(p0 + o1)));
    const __m128i v2 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)(p0 + o2)));
    const __m128i v3 = _mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)(p0 + S)));
    __m128i acc = _mm_slli_epi32(v0, 12);
    acc = _mm_add_epi32(acc, _mm_mullo_epi32(_mm_sub_epi32(v1, v0), _mm_set1_epi32(wa)));
    acc = _mm_add_epi32(acc, _mm_mullo_epi32(_mm_sub_epi32(v2, v1), _mm_set1_epi32(wb)));
    acc = _mm_add_epi32(acc, _mm_mullo_epi32(_mm_sub_epi32(v3, v2), _mm_set1_epi32(wc)));
    acc = _mm_srai_epi32(_mm_add_epi32(acc, rnd), 20);
    const __m128i p16 = _mm_packs_epi32(acc, acc);
    const __m128i p8 = _mm_packus_epi16(p16, p16);
    storePix<F>(d, x, (uint32_t)_mm_cvtsi128_si32(p8));
  }
}
#endif

template <OutFormat F>
HIE_AI void trilinearRow(const LutRT& L, const uint8_t* r, const uint8_t* g, const uint8_t* b,
                         int w, uint8_t* d) {
  const uint16_t* nd = L.nodes.data();
  const uint32_t* oR = L.offR.data();
  const uint32_t* oG = L.offG.data();
  const uint32_t* oB = L.offB.data();
  const uint16_t* ft = L.frac;
  const int sR = (int)L.sR, sG = (int)L.sG, sB = (int)L.sB;
  for (int x = 0; x < w; ++x) {
    const int R = r[x], G = g[x], B = b[x];
    const uint16_t* p = nd + oR[R] + oG[G] + oB[B];
    const int fr = ft[R], fg = ft[G], fb = ft[B];
    int out[3];
    for (int c = 0; c < 3; ++c) {
      int c000 = p[c], c100 = p[sR + c], c010 = p[sG + c], c110 = p[sR + sG + c];
      int c001 = p[sB + c], c101 = p[sR + sB + c], c011 = p[sG + sB + c], c111 = p[sR + sG + sB + c];
      int a00 = (c000 * (4096 - fr) + c100 * fr + 2048) >> 12;
      int a10 = (c010 * (4096 - fr) + c110 * fr + 2048) >> 12;
      int a01 = (c001 * (4096 - fr) + c101 * fr + 2048) >> 12;
      int a11 = (c011 * (4096 - fr) + c111 * fr + 2048) >> 12;
      int b0 = (a00 * (4096 - fg) + a10 * fg + 2048) >> 12;
      int b1 = (a01 * (4096 - fg) + a11 * fg + 2048) >> 12;
      out[c] = (b0 * (4096 - fb) + b1 * fb + (1 << 19)) >> 20;
    }
    store<F>(d, x, out[0], out[1], out[2]);
  }
}

}  // namespace

void lutApplyRow(const LutRT& L, bool tetra, const uint8_t* r, const uint8_t* g,
                 const uint8_t* b, int w, OutFormat fmt, uint8_t* d) {
#ifdef HIE_SSE41_PATH
  static const bool hasSse41 = __builtin_cpu_supports("sse4.1");
  const bool sse = hasSse41 && !g_forceScalar.load(std::memory_order_relaxed);
#else
  const bool sse = false;
#endif
  (void)sse;
#ifdef HIE_SSE41_PATH
#define HIE_TETRA(F) (sse ? tetraRowSse41<OutFormat::F>(L, r, g, b, w, d) : tetraRowScalar<OutFormat::F>(L, r, g, b, w, d))
#else
#define HIE_TETRA(F) tetraRowScalar<OutFormat::F>(L, r, g, b, w, d)
#endif
#define HIE_CASE(F)                                                         \
  case OutFormat::F:                                                        \
    if (tetra) HIE_TETRA(F);                                                \
    else trilinearRow<OutFormat::F>(L, r, g, b, w, d);                      \
    break;
  switch (fmt) {
    HIE_CASE(RGB24)
    HIE_CASE(BGR24)
    HIE_CASE(RGBA32)
    HIE_CASE(BGRA32)
  }
#undef HIE_CASE
#undef HIE_TETRA
}

}  // namespace hie
