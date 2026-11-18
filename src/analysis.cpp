#include "analysis.hpp"

#include <cstring>

namespace hie {
namespace {

struct Tables {
  float lin[256];
  Tables() {
    for (int i = 0; i < 256; ++i) lin[i] = srgbToLin(i / 255.0f);
  }
};

float percentile(const int* hist, int bins, int total, float q) {
  long target = (long)(q * (float)total);
  long cum = 0;
  for (int i = 0; i < bins; ++i) {
    cum += hist[i];
    if (cum > target) return (float)i;
  }
  return (float)(bins - 1);
}

}  // namespace

float estimateNoiseSigma(const uint8_t* c, int n) {
  if (!c || n < 8) return 0.f;
  int hist[1024];
  std::memset(hist, 0, sizeof(hist));
  long total = 0;
  for (int y = 1; y < n - 1; ++y) {
    const uint8_t* a = c + (size_t)(y - 1) * n;
    const uint8_t* b = c + (size_t)y * n;
    const uint8_t* d = c + (size_t)(y + 1) * n;
    for (int x = 1; x < n - 1; ++x) {
      int r = a[x - 1] - 2 * a[x] + a[x + 1] - 2 * b[x - 1] + 4 * b[x] - 2 * b[x + 1] + d[x - 1] -
              2 * d[x] + d[x + 1];
      if (r < 0) r = -r;
      if (r > 1023) r = 1023;
      ++hist[r];
      ++total;
    }
  }
  float med = percentile(hist, 1024, (int)total, 0.5f) + 0.5f;
  // response of the 3x3 Laplacian-difference kernel to white noise has std = 6*sigma
  return med / (0.6745f * 6.0f);
}

void analyzeThumbnail(const Thumbnail& t, const Config& cfg, Params& out, AnalysisScratch& sc) {
  static const Tables T;
  out = cfg.manual;
  if (!t.rgb) return;
  const int N = kThumbSize * kThumbSize;
  const uint8_t* p = t.rgb;

  int hl[256], hm[256];  // display luma / max-channel histograms
  std::memset(hl, 0, sizeof(hl));
  std::memset(hm, 0, sizeof(hm));
  double chroma_sum = 0;
  int chroma_n = 0;
  for (int i = 0; i < N; ++i) {
    int r = p[3 * i], g = p[3 * i + 1], b = p[3 * i + 2];
    int y = (54 * r + 183 * g + 19 * b + 128) >> 8;
    int mx = std::max(r, std::max(g, b)), mn = std::min(r, std::min(g, b));
    ++hl[y];
    ++hm[mx];
    if (y >= 30 && y <= 235) {
      chroma_sum += (mx - mn);
      ++chroma_n;
    }
  }
  const float p1 = percentile(hl, 256, N, 0.01f), p5 = percentile(hl, 256, N, 0.05f);
  const float p50 = percentile(hl, 256, N, 0.50f), p95 = percentile(hl, 256, N, 0.95f);
  const float p99 = percentile(hl, 256, N, 0.99f);
  const float m99 = percentile(hm, 256, N, 0.99f);

  // ---- white balance: gray-edge illuminant estimate, gated by a neutral-pixel check ---------------
  // Gray-world is fooled by big uniform areas (sky, grass, walls). Gray-edge (van de Weijer 2007)
  // uses the mean *edge* colour instead, which those areas do not contribute to. The estimate is
  // only trusted when applying it measurably increases the share of near-neutral pixels
  // (what a correct white balance does when neutral surfaces exist); otherwise confidence is 0 and
  // the image is left alone ("do no harm").
  float gr = 1.f, gb = 1.f, conf = 0.f;
  if (cfg.auto_wb) {
    constexpr int G = kThumbSize / 2;                 // 128x128 working grid, linear light
    float* L = sc.lin.data();                         // [3][G*G]
    for (int y = 0; y < G; ++y)
      for (int x = 0; x < G; ++x)
        for (int c = 0; c < 3; ++c) {
          const uint8_t* q = p + ((size_t)(2 * y) * kThumbSize + 2 * x) * 3 + c;
          L[c * G * G + y * G + x] = 0.25f * (T.lin[q[0]] + T.lin[q[3]] + T.lin[q[kThumbSize * 3]] + T.lin[q[kThumbSize * 3 + 3]]);
        }
    double e[3] = {0, 0, 0};
    long edges = 0;
    for (int y = 1; y < G - 1; ++y)
      for (int x = 1; x < G - 1; ++x) {
        const int i = y * G + x;
        float yc = 0.2126f * L[i] + 0.7152f * L[G * G + i] + 0.0722f * L[2 * G * G + i];
        float yx = 0.2126f * (L[i + 1] - L[i - 1]) + 0.7152f * (L[G * G + i + 1] - L[G * G + i - 1]) + 0.0722f * (L[2 * G * G + i + 1] - L[2 * G * G + i - 1]);
        float yy = 0.2126f * (L[i + G] - L[i - G]) + 0.7152f * (L[G * G + i + G] - L[G * G + i - G]) + 0.0722f * (L[2 * G * G + i + G] - L[2 * G * G + i - G]);
        if (std::fabs(yx) + std::fabs(yy) < 0.12f * (yc + 0.02f)) continue;     // not an edge
        if (yc < 0.004f || yc > 0.9f) continue;
        for (int c = 0; c < 3; ++c) {
          const float* ch = L + c * G * G + i;
          e[c] += std::fabs(ch[1] - ch[-1]) + std::fabs(ch[G] - ch[-G]);
        }
        ++edges;
      }
    // neutral-pixel share under candidate gains, on a strided subset of the 128x128 grid
    auto neutralShare = [&](float rg, float bg, int stride, double* mean) {
      double sr = 0, sg = 0, sb = 0; long n = 0, tot = 0;
      for (int y = 0; y < G; y += stride)
        for (int x = 0; x < G; x += stride) {
          const int i = y * G + x;
          float lr = L[i] * rg, lg = L[G * G + i], lb = L[2 * G * G + i] * bg;
          float yv = 0.2126f * lr + 0.7152f * lg + 0.0722f * lb;
          if (yv < 0.01f || yv > 0.85f) continue;
          ++tot;
          float m = (lr + lg + lb) * (1.f / 3.f), tol = 0.14f * m;
          if (std::fabs(lr - m) > tol || std::fabs(lg - m) > tol || std::fabs(lb - m) > tol) continue;
          sr += lr; sg += lg; sb += lb; ++n;
        }
      if (mean) { mean[0] = sr; mean[1] = sg; mean[2] = sb; }
      return tot ? (float)n / (float)tot : 0.f;
    };
    // Gray-edge estimate, refined once on the pixels it makes neutral, then accepted only if it
    // multiplies the neutral share (a real cast hides neutrals; a cast-free scene already has them).
    const float n0 = neutralShare(1.f, 1.f, 1, nullptr);
    float n1 = n0;
    if (edges >= 100 && e[0] > 0 && e[2] > 0) {
      float cr = std::min(std::max((float)(e[1] / e[0]), 0.5f), 2.0f);
      float cb = std::min(std::max((float)(e[1] / e[2]), 0.5f), 2.0f);
      double m1[3];
      n1 = neutralShare(cr, cb, 1, m1);
      if (m1[0] > 0 && m1[2] > 0) {
        float rr = std::min(std::max(cr * (float)(m1[1] / m1[0]), 0.5f), 2.0f);
        float rb = std::min(std::max(cb * (float)(m1[1] / m1[2]), 0.5f), 2.0f);
        float n2 = neutralShare(rr, rb, 1, nullptr);
        if (n2 >= n1) { cr = rr; cb = rb; n1 = n2; }
      }
      gr = cr; gb = cb;
      conf = n1 < 0.02f ? 0.f : std::min(std::max((n1 / std::max(n0, 0.02f) - 1.25f) / 1.5f, 0.f), 1.f);
    }
    if (conf <= 0.f) gr = gb = 1.f;
  }
  if (cfg.auto_wb) {
    float s = cfg.wb_strength * conf;
    out.wb_r = std::min(std::max(std::exp(std::log(gr) * s), 0.67f), 1.5f);
    out.wb_b = std::min(std::max(std::exp(std::log(gb) * s), 0.67f), 1.5f);
  }

  // ---- exposure --------------------------------------------------------------------------------
  if (cfg.auto_exposure) {
    float m_lin = std::max(T.lin[(int)p50], 1e-3f);
    float tgt = srgbToLin(cfg.target_median);
    float ev = std::log2(tgt / m_lin);
    const float dead = 0.2f;
    ev = (std::fabs(ev) <= dead) ? 0.f : (ev > 0 ? ev - dead : ev + dead);
    ev *= cfg.exposure_strength;
    ev = std::min(std::max(ev, -1.0f), 1.5f);
    // keep the bright end from clipping hard (account for white-balance gain on the max channel)
    float wsum = 0.2126f * out.wb_r + 0.7152f + 0.0722f * out.wb_b;
    float gmax = std::max(out.wb_r, std::max(1.f, out.wb_b)) / wsum;
    float ev_cap = std::log2(1.0f / std::max(T.lin[(int)m99], 1e-3f)) + 0.35f - std::log2(gmax);
    if (ev > std::max(ev_cap, 0.f)) ev = std::max(ev_cap, 0.f);
    out.exposure_ev = ev;
  }

  // ---- contrast / saturation / shadows ----------------------------------------------------------
  const float spread = (p95 - p5) / 255.f;
  if (cfg.auto_contrast) {
    // only ever add contrast to flat (washed-out) scenes
    out.contrast = std::min(std::max(1.f + 0.9f * (0.72f - spread), 1.f), 1.35f);
  }
  if (cfg.auto_saturation) {
    float mc = chroma_n ? (float)(chroma_sum / chroma_n) / 255.f : 0.2f;
    out.saturation = std::min(std::max(1.f + 0.5f * (0.20f - mc) / 0.20f, 1.f), 1.25f);
  }
  if (cfg.auto_shadows) {
    float dark = 0;
    for (int i = 0; i <= 30; ++i) dark += (float)hl[i];
    dark /= (float)N;
    out.shadow_lift = std::min(std::max(1.2f * (dark - 0.20f), 0.f), 0.5f);
  }

  // ---- noise-aware local processing ---------------------------------------------------------------
  const float sigma = estimateNoiseSigma(t.luma_crop, kCropSize);
  if (cfg.auto_denoise) {
    out.denoise_t = sigma < 0.9f ? 0.f : std::min(std::max(3.2f * sigma, 3.f), 40.f);
  }
  if (cfg.auto_clahe) {
    float range_d = (p99 - p1) / 255.f;
    float need = std::min(std::max((0.92f - range_d) / 0.45f, 0.2f), 1.0f);
    float noisy = std::min(std::max(1.f - (sigma - 2.f) / 10.f, 0.4f), 1.0f);
    out.clahe_strength = cfg.clahe_strength_max * need * noisy;
    out.clahe_clip = std::min(std::max(cfg.clahe_clip_base - 0.12f * std::max(sigma - 1.f, 0.f), 1.5f),
                              cfg.clahe_clip_base);
  }
}

}  // namespace hie
