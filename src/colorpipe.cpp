#include "colorpipe.hpp"

#include <cmath>

#include "lut3d.hpp"

namespace hie {

namespace {
constexpr uint32_t kWR = 13933, kWB = 4732, kWG = 65536 - kWR - kWB;   // Rec.709 luma, Q16
}

void FactoredRT::build(const Params& p) {
  identity = lutIsNeutral(p);
  if (identity) return;
  float gr = p.wb_r, gg = 1.f, gb = p.wb_b;
  float norm = 0.2126f * gr + 0.7152f * gg + 0.0722f * gb;
  float ev = std::exp2(p.exposure_ev) / norm;
  const float gain[3] = {gr * ev, gg * ev, gb * ev};
  for (int c = 0; c < 3; ++c)
    for (int v = 0; v < 256; ++v) {
      float lin = std::min(srgbToLin(v / 255.f) * gain[c], 1.f);
      T[c][v] = (int32_t)std::lround(linToSrgb(lin) * 65535.f);
    }
  for (int i = 0; i < 4096; ++i) {
    float yd = ((float)i + 0.5f) / 4096.f;
    float yt = toneCurveShared(yd, p.shadow_lift, p.contrast);
    float k1 = p.saturation * (yt / yd);
    float k0 = (1.f - p.saturation) * yt;
    int a = (int)std::lround(k1 * 4096.f);
    a = a < 0 ? 0 : (a > 32767 ? 32767 : a);
    int b = (int)std::lround(k0 * 32768.f);
    b = b < -32768 ? -32768 : (b > 32767 ? 32767 : b);
    K[i] = (uint32_t)a | ((uint32_t)(uint16_t)(int16_t)b << 16);
  }
}

// Note: AVX2 gather variants were measured slower than this scalar loop on Intel parts with the
// Downfall (GDS) microcode mitigation, and are not portable to NEON anyway.
HIE_MV static void factoredRowScalar(const FactoredRT& f, uint8_t* HIE_RESTRICT r, uint8_t* HIE_RESTRICT g,
                        uint8_t* HIE_RESTRICT b, int w) {
  const int32_t* HIE_RESTRICT T0 = f.T[0];
  const int32_t* HIE_RESTRICT T1 = f.T[1];
  const int32_t* HIE_RESTRICT T2 = f.T[2];
  const uint32_t* HIE_RESTRICT K = f.K;
  for (int x = 0; x < w; ++x) {
    const uint32_t dr = (uint32_t)T0[r[x]], dg = (uint32_t)T1[g[x]], db = (uint32_t)T2[b[x]];
    const uint32_t y = (kWR * dr + kWG * dg + kWB * db) >> 16;
    const uint32_t kk = K[y >> 4];
    const int32_t k1 = (int32_t)(kk & 0xFFFFu);
    const int32_t k0 = ((int32_t)kk >> 16) * 2;   // arithmetic shift -> signed, Q15 -> Q16
    int32_t o0 = ((k1 * (int32_t)dr) >> 12) + k0;
    int32_t o1 = ((k1 * (int32_t)dg) >> 12) + k0;
    int32_t o2 = ((k1 * (int32_t)db) >> 12) + k0;
    o0 = o0 < 0 ? 0 : (o0 > 65535 ? 65535 : o0);
    o1 = o1 < 0 ? 0 : (o1 > 65535 ? 65535 : o1);
    o2 = o2 < 0 ? 0 : (o2 > 65535 ? 65535 : o2);
    r[x] = (uint8_t)((o0 * 255 + 32768) >> 16);
    g[x] = (uint8_t)((o1 * 255 + 32768) >> 16);
    b[x] = (uint8_t)((o2 * 255 + 32768) >> 16);
  }
}

void factoredRow(const FactoredRT& f, uint8_t* r, uint8_t* g, uint8_t* b, int w) {
  factoredRowScalar(f, r, g, b, w);
}

}  // namespace hie
