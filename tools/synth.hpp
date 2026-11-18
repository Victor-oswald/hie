// Synthetic test scenes and RGB -> YUV frame packers (float reference path, independent of the
// engine's fixed-point code). Used by hie_cli bench and the test-suite.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "hie/hie.h"

namespace synth {

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed = 1) : s(seed * 2685821657736338717ull + 88172645463325252ull) {}
  uint32_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 11); }
  float uni() { return (next() & 0xFFFFFF) * (1.0f / 16777216.0f); }
  float gauss() {
    float u1 = std::max(uni(), 1e-7f), u2 = uni();
    return std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
  }
};

inline float s2l(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }
inline float l2s(float c) {
  c = std::min(std::max(c, 0.f), 1.f);
  return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.f / 2.4f) - 0.055f;
}

// Photographic-ish scene: smooth colour field, coloured blocks, neutral step wedge, edges.
// exposure & cast act in linear light; noise is additive Gaussian in 8-bit levels.
inline std::vector<uint8_t> sceneRGB(int w, int h, uint64_t seed, float exposure = 1.f,
                                     const float* cast = nullptr, float noise = 0.f) {
  std::vector<uint8_t> o((size_t)w * h * 3);
  Rng rng(seed);
  const float c0[3] = {cast ? cast[0] : 1.f, cast ? cast[1] : 1.f, cast ? cast[2] : 1.f};
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float u = (float)x / w, v = (float)y / h;
      float rgb[3] = {0.50f + 0.30f * std::sin(6.2831853f * (u + 0.1f * v)),
                      0.48f + 0.30f * std::sin(6.2831853f * (1.2f * v + 0.3f * u + 0.33f)),
                      0.46f + 0.30f * std::sin(6.2831853f * (0.7f * u + 0.5f * v + 0.66f))};
      // neutral step wedge along the bottom
      if (v > 0.82f) {
        float g = std::floor(u * 8.f) / 7.f;
        rgb[0] = rgb[1] = rgb[2] = 0.05f + 0.9f * g;
      }
      // saturated blocks
      if (v > 0.1f && v < 0.3f) {
        int k = (int)(u * 6.f);
        static const float blk[6][3] = {{0.8f, 0.2f, 0.15f}, {0.2f, 0.65f, 0.25f}, {0.15f, 0.25f, 0.8f},
                                        {0.9f, 0.8f, 0.15f}, {0.75f, 0.25f, 0.7f}, {0.2f, 0.7f, 0.75f}};
        if (std::fmod(u * 6.f, 1.f) > 0.08f)
          for (int c = 0; c < 3; ++c) rgb[c] = blk[k % 6][c];
      }
      for (int c = 0; c < 3; ++c) {
        float lin = s2l(std::min(std::max(rgb[c], 0.f), 1.f)) * exposure * c0[c];
        float d = l2s(lin) * 255.f;
        if (noise > 0.f) d += noise * rng.gauss();
        o[((size_t)y * w + x) * 3 + c] = (uint8_t)std::lround(std::min(std::max(d, 0.f), 255.f));
      }
    }
  return o;
}

// Float reference RGB24 -> {I420, NV12, YUYV, RGB24}. Chroma is box-averaged (4:2:0 / 4:2:2).
inline std::vector<uint8_t> rgbToFormat(const std::vector<uint8_t>& rgb, int w, int h,
                                        hie::InFormat f, hie::Matrix m, hie::Range r) {
  if (f == hie::InFormat::RGB24) return rgb;
  const double kr = m == hie::Matrix::BT601 ? 0.299 : 0.2126, kb = m == hie::Matrix::BT601 ? 0.114 : 0.0722;
  const double kg = 1.0 - kr - kb;
  const bool lim = r == hie::Range::Limited;
  const double ys = lim ? 219.0 : 255.0, cs = lim ? 224.0 : 255.0, yo = lim ? 16.0 : 0.0;
  std::vector<float> Y((size_t)w * h), U((size_t)w * h), V((size_t)w * h);
  for (size_t i = 0; i < (size_t)w * h; ++i) {
    double R = rgb[3 * i], G = rgb[3 * i + 1], B = rgb[3 * i + 2];
    double yf = kr * R + kg * G + kb * B;
    Y[i] = (float)(yo + ys * yf / 255.0);
    U[i] = (float)(128.0 + cs * (B - yf) / (2.0 * (1.0 - kb)) / 255.0);
    V[i] = (float)(128.0 + cs * (R - yf) / (2.0 * (1.0 - kr)) / 255.0);
  }
  auto q = [](float v) { return (uint8_t)std::lround(std::min(std::max(v, 0.f), 255.f)); };
  const int cw = (w + 1) / 2, ch = (h + 1) / 2;
  auto avg = [&](const std::vector<float>& P, int cx, int cy, int sy) {
    float s = 0; int n = 0;
    for (int dy = 0; dy < sy; ++dy)
      for (int dx = 0; dx < 2; ++dx) {
        int x = std::min(2 * cx + dx, w - 1), y = std::min(sy * cy + dy, h - 1);
        s += P[(size_t)y * w + x]; ++n;
      }
    return s / n;
  };
  std::vector<uint8_t> o;
  if (f == hie::InFormat::I420) {
    o.resize((size_t)w * h + 2 * (size_t)cw * ch);
    for (size_t i = 0; i < (size_t)w * h; ++i) o[i] = q(Y[i]);
    uint8_t* pu = o.data() + (size_t)w * h; uint8_t* pv = pu + (size_t)cw * ch;
    for (int cy = 0; cy < ch; ++cy)
      for (int cx = 0; cx < cw; ++cx) { pu[(size_t)cy * cw + cx] = q(avg(U, cx, cy, 2)); pv[(size_t)cy * cw + cx] = q(avg(V, cx, cy, 2)); }
  } else if (f == hie::InFormat::NV12) {
    o.resize((size_t)w * h + (size_t)cw * 2 * ch);
    for (size_t i = 0; i < (size_t)w * h; ++i) o[i] = q(Y[i]);
    uint8_t* uv = o.data() + (size_t)w * h;
    for (int cy = 0; cy < ch; ++cy)
      for (int cx = 0; cx < cw; ++cx) { uv[(size_t)cy * cw * 2 + 2 * cx] = q(avg(U, cx, cy, 2)); uv[(size_t)cy * cw * 2 + 2 * cx + 1] = q(avg(V, cx, cy, 2)); }
  } else {  // YUYV 4:2:2
    o.resize((size_t)cw * 4 * h);
    for (int y = 0; y < h; ++y)
      for (int cx = 0; cx < cw; ++cx) {
        uint8_t* d = o.data() + (size_t)y * cw * 4 + cx * 4;
        d[0] = q(Y[(size_t)y * w + std::min(2 * cx, w - 1)]);
        d[1] = q(avg(U, cx, y, 1));
        d[2] = q(Y[(size_t)y * w + std::min(2 * cx + 1, w - 1)]);
        d[3] = q(avg(V, cx, y, 1));
      }
  }
  return o;
}

}  // namespace synth
