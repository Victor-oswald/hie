#include "kernels.hpp"

#include <cmath>
#include <cstring>

namespace hie {
namespace k {

// ---------------------------------------------------------------------------------------------
// ingest
// ---------------------------------------------------------------------------------------------
HIE_MV void yuyvExtractY(const uint8_t* HIE_RESTRICT src, int ss, uint8_t* HIE_RESTRICT dst, int ds,
                         int w, int y0, int y1) {
  for (int y = y0; y < y1; ++y) {
    const uint8_t* s = src + (size_t)y * ss;
    uint8_t* d = dst + (size_t)y * ds;
    for (int x = 0; x < w; ++x) d[x] = s[2 * x];
  }
}

HIE_MV void rgbToLuma(const uint8_t* HIE_RESTRICT src, int ss, uint8_t* HIE_RESTRICT dst, int ds,
                      int w, int y0, int y1, const YuvCoeffs& c) {
  const int fr = c.fyr, fg = c.fyg, fb = c.fyb, lo = c.ylo;
  for (int y = y0; y < y1; ++y) {
    const uint8_t* s = src + (size_t)y * ss;
    uint8_t* d = dst + (size_t)y * ds;
    for (int x = 0; x < w; ++x) {
      int v = lo + ((fr * s[3 * x] + fg * s[3 * x + 1] + fb * s[3 * x + 2] + 8192) >> 14);
      d[x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
  }
}

// Constant strides (NV12 = 2, YUYV = 4) so the compiler can vectorise the de-interleave.
HIE_MV void gatherStrided(const uint8_t* HIE_RESTRICT src, int step, int n, uint8_t* HIE_RESTRICT dst) {
  if (step == 2) {
    for (int i = 0; i < n; ++i) dst[i] = src[2 * i];
  } else if (step == 4) {
    for (int i = 0; i < n; ++i) dst[i] = src[4 * i];
  } else {
    for (int i = 0; i < n; ++i) dst[i] = src[(size_t)i * step];
  }
}

// ---------------------------------------------------------------------------------------------
// denoise
// ---------------------------------------------------------------------------------------------
static inline int bilatPixel(const uint8_t* a, const uint8_t* b, const uint8_t* c, int x, int w,
                             int T) {
  const int xm = x > 0 ? x - 1 : 0, xp = x < w - 1 ? x + 1 : w - 1;
  const int c0 = b[x];
  const int n[8] = {a[xm], a[x], a[xp], b[xm], b[xp], c[xm], c[x], c[xp]};
  static const int sw[8] = {1, 2, 1, 2, 2, 1, 2, 1};
  int ws = 4 * T, ds = 0;
  for (int i = 0; i < 8; ++i) {
    int d = n[i] - c0;
    int r = T - (d < 0 ? -d : d);
    r = r < 0 ? 0 : r;
    int wt = r * sw[i];
    ws += wt;
    ds += wt * d;
  }
  float q = (float)ds / (float)ws;
  return c0 + (int)std::floor(q + 0.5f);
}

HIE_MV void denoiseRows(const uint8_t* HIE_RESTRICT src, int ss, uint8_t* HIE_RESTRICT dst, int ds,
                        int w, int h, int y0, int y1, int T) {
  for (int y = y0; y < y1; ++y) {
    const uint8_t* b = src + (size_t)y * ss;
    const uint8_t* a = src + (size_t)(y > 0 ? y - 1 : 0) * ss;
    const uint8_t* c = src + (size_t)(y < h - 1 ? y + 1 : h - 1) * ss;
    uint8_t* o = dst + (size_t)y * ds;
    o[0] = (uint8_t)bilatPixel(a, b, c, 0, w, T);
    if (w > 1) o[w - 1] = (uint8_t)bilatPixel(a, b, c, w - 1, w, T);
    for (int x = 1; x < w - 1; ++x) {
      const int c0 = b[x];
      const int n0 = a[x - 1], n1 = a[x], n2 = a[x + 1];
      const int n3 = b[x - 1], n4 = b[x + 1];
      const int n5 = c[x - 1], n6 = c[x], n7 = c[x + 1];
      int d0 = n0 - c0, d1 = n1 - c0, d2 = n2 - c0, d3 = n3 - c0;
      int d4 = n4 - c0, d5 = n5 - c0, d6 = n6 - c0, d7 = n7 - c0;
      int r0 = T - (d0 < 0 ? -d0 : d0); r0 = r0 < 0 ? 0 : r0;
      int r1 = T - (d1 < 0 ? -d1 : d1); r1 = r1 < 0 ? 0 : r1;
      int r2 = T - (d2 < 0 ? -d2 : d2); r2 = r2 < 0 ? 0 : r2;
      int r3 = T - (d3 < 0 ? -d3 : d3); r3 = r3 < 0 ? 0 : r3;
      int r4 = T - (d4 < 0 ? -d4 : d4); r4 = r4 < 0 ? 0 : r4;
      int r5 = T - (d5 < 0 ? -d5 : d5); r5 = r5 < 0 ? 0 : r5;
      int r6 = T - (d6 < 0 ? -d6 : d6); r6 = r6 < 0 ? 0 : r6;
      int r7 = T - (d7 < 0 ? -d7 : d7); r7 = r7 < 0 ? 0 : r7;
      int ws = 4 * T + r0 + 2 * r1 + r2 + 2 * r3 + 2 * r4 + r5 + 2 * r6 + r7;
      int dsum = r0 * d0 + 2 * r1 * d1 + r2 * d2 + 2 * r3 * d3 + 2 * r4 * d4 + r5 * d5 +
                 2 * r6 * d6 + r7 * d7;
      // round-to-nearest of dsum/ws without floor(): |q| < 256, so bias to positive and truncate
      float q = (float)dsum / (float)ws;
      int v = c0 + ((int)(q + 256.5f) - 256);
      o[x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
  }
}

// ---------------------------------------------------------------------------------------------
// CLAHE
// ---------------------------------------------------------------------------------------------
void claheTileLut(const uint8_t* y, int stride, int x0, int y0, int x1, int y1, int hs, float clip,
                  float strength, int lo, int hi, uint16_t* lut) {
  uint32_t h[256];
  std::memset(h, 0, sizeof(h));
  uint32_t n = 0;
  for (int yy = y0; yy < y1; yy += hs) {
    const uint8_t* row = y + (size_t)yy * stride;
    for (int xx = x0; xx < x1; xx += hs) {
      ++h[row[xx]];
      ++n;
    }
  }
  if (n == 0) {
    for (int i = 0; i < 256; ++i) lut[i] = (uint16_t)(i * 256);
    return;
  }
  uint32_t limit = (uint32_t)(clip * (float)n / 256.0f);
  if (limit < 1) limit = 1;
  uint32_t excess = 0;
  for (int i = 0; i < 256; ++i)
    if (h[i] > limit) {
      excess += h[i] - limit;
      h[i] = limit;
    }
  uint32_t add = excess / 256, rem = excess % 256;
  for (int i = 0; i < 256; ++i) h[i] += add;
  for (uint32_t i = 0; i < rem; ++i) ++h[(i * 256u) / rem];

  const float span = (float)(hi - lo);
  const float inv_n = 1.0f / (float)n;
  uint32_t cum = 0;
  int prev = 0;
  for (int i = 0; i < 256; ++i) {
    float mid = ((float)cum + 0.5f * (float)h[i]) * inv_n;
    cum += h[i];
    float m = (float)lo + mid * span;
    float v = (1.0f - strength) * (float)i + strength * m;
    if (v < (float)lo) v = (float)lo;
    if (v > (float)hi) v = (float)hi;
    int q = (int)(v * 256.0f + 0.5f);
    if (q < prev) q = prev;  // monotone
    prev = q;
    lut[i] = (uint16_t)q;
  }
}

HIE_MV void claheVBlend(const uint16_t* HIE_RESTRICT a, const uint16_t* HIE_RESTRICT b, int fy,
                        uint16_t* HIE_RESTRICT out, int n) {
  const int f0 = 256 - fy;
  for (int i = 0; i < n; ++i) out[i] = (uint16_t)(((int)a[i] * f0 + (int)b[i] * fy + 128) >> 8);
}

void claheApplyRow(const uint8_t* yin, uint8_t* yout, const ClaheSeg* segs, int nseg,
                   const uint16_t* fx, const uint16_t* rowLut) {
  for (int s = 0; s < nseg; ++s) {
    const uint16_t* L0 = rowLut + (size_t)segs[s].t0 * 256;
    const uint16_t* L1 = rowLut + (size_t)segs[s].t1 * 256;
    const int x1 = segs[s].x1;
    for (int x = segs[s].x0; x < x1; ++x) {
      const int v = yin[x];
      const int f = fx[x];
      yout[x] = (uint8_t)(((int)L0[v] * (256 - f) + (int)L1[v] * f + 32768) >> 16);
    }
  }
}

// ---------------------------------------------------------------------------------------------
// chroma / colour
// ---------------------------------------------------------------------------------------------
HIE_MV void vblend31(const uint8_t* HIE_RESTRICT a, const uint8_t* HIE_RESTRICT b,
                     uint8_t* HIE_RESTRICT o, int n) {
  for (int i = 0; i < n; ++i) o[i] = (uint8_t)((3 * (int)a[i] + (int)b[i] + 2) >> 2);
}

HIE_MV void hupLinear(const uint8_t* HIE_RESTRICT c, int cw, uint8_t* HIE_RESTRICT o, int w) {
  // centre-sited chroma (JPEG/MPEG-1 style): out[2i] = (3c[i]+c[i-1])/4, out[2i+1] = (3c[i]+c[i+1])/4
  const int last = cw - 1;
  {
    o[0] = c[0];
    if (w > 1) o[1] = (uint8_t)((3 * (int)c[0] + (int)c[last > 0 ? 1 : 0] + 2) >> 2);
  }
  for (int i = 1; i < last; ++i) {
    int ci = 3 * (int)c[i] + 2;
    o[2 * i] = (uint8_t)((ci + (int)c[i - 1]) >> 2);
    o[2 * i + 1] = (uint8_t)((ci + (int)c[i + 1]) >> 2);
  }
  if (last > 0) {
    int i = last;
    o[2 * i] = (uint8_t)((3 * (int)c[i] + (int)c[i - 1] + 2) >> 2);
    if (2 * i + 1 < w) o[2 * i + 1] = c[i];
  }
}

HIE_MV void hupNearest(const uint8_t* HIE_RESTRICT c, uint8_t* HIE_RESTRICT o, int w) {
  for (int x = 0; x < w; ++x) o[x] = c[x >> 1];
}

HIE_MV void yuvToRgb(const uint8_t* HIE_RESTRICT Y, const uint8_t* HIE_RESTRICT U,
                     const uint8_t* HIE_RESTRICT V, int w, const YuvCoeffs& c,
                     uint8_t* HIE_RESTRICT R, uint8_t* HIE_RESTRICT G, uint8_t* HIE_RESTRICT B) {
  const int ylo = c.ylo, cy = c.cy, crv = c.crv, cgu = c.cgu, cgv = c.cgv, cbu = c.cbu;
  for (int x = 0; x < w; ++x) {
    int yy = ((int)Y[x] - ylo) * cy + 8192;
    int u = (int)U[x] - 128, v = (int)V[x] - 128;
    int r = (yy + crv * v) >> 14;
    int g = (yy - cgu * u - cgv * v) >> 14;
    int b = (yy + cbu * u) >> 14;
    R[x] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
    G[x] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
    B[x] = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b));
  }
}

HIE_MV void rgbApplyLumaDelta(const uint8_t* HIE_RESTRICT rgb, const uint8_t* HIE_RESTRICT yE,
                              const uint8_t* HIE_RESTRICT yO, int w, int cy, uint8_t* HIE_RESTRICT R,
                              uint8_t* HIE_RESTRICT G, uint8_t* HIE_RESTRICT B) {
  for (int x = 0; x < w; ++x) {
    int d = (((int)yE[x] - (int)yO[x]) * cy + 8192) >> 14;
    int r = (int)rgb[3 * x] + d, g = (int)rgb[3 * x + 1] + d, b = (int)rgb[3 * x + 2] + d;
    R[x] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
    G[x] = (uint8_t)(g < 0 ? 0 : (g > 255 ? 255 : g));
    B[x] = (uint8_t)(b < 0 ? 0 : (b > 255 ? 255 : b));
  }
}

}  // namespace k
}  // namespace hie
