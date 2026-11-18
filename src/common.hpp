// Internal helpers: SIMD dispatch macro, colour-space coefficients, sRGB transfer functions.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "hie/hie.h"

// Runtime ISA dispatch on x86 (AVX2 / SSE4.1 / baseline) via function multiversioning.
// On ARM the compiler targets NEON by default; MSVC builds should pass /arch:AVX2 (see CMake).
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__APPLE__) && !defined(_WIN32)
#define HIE_MV __attribute__((target_clones("avx2", "sse4.1", "default")))
#else
#define HIE_MV
#endif

#if defined(_MSC_VER)
#define HIE_AI __forceinline
#define HIE_RESTRICT __restrict
#else
#define HIE_AI __attribute__((always_inline)) inline
#define HIE_RESTRICT __restrict__
#endif

namespace hie {

// Fixed-point (Q14) YCbCr <-> RGB coefficients for a matrix/range combination.
struct YuvCoeffs {
  int ylo;                       // luma black level (16 / 0)
  int ymin, ymax;                // legal luma interval (16..235 / 0..255)
  int cy, crv, cgu, cgv, cbu;    // inverse: R = cy*(Y-ylo) + crv*V' ...
  int fyr, fyg, fyb;             // forward: Y = ylo + (fyr*R + fyg*G + fyb*B)
  float y_gain;                  // d(RGB)/d(Y) = cy / 16384
};

inline YuvCoeffs makeCoeffs(Matrix m, Range r) {
  const double kr = (m == Matrix::BT601) ? 0.299 : 0.2126;
  const double kb = (m == Matrix::BT601) ? 0.114 : 0.0722;
  const double kg = 1.0 - kr - kb;
  const bool lim = (r == Range::Limited);
  const double yscale = lim ? 219.0 : 255.0, cscale = lim ? 224.0 : 255.0;
  YuvCoeffs c{};
  c.ylo = lim ? 16 : 0;
  c.ymin = lim ? 16 : 0;
  c.ymax = lim ? 235 : 255;
  const double Q = 16384.0;
  // inverse: 8-bit code values in, 8-bit RGB out  ->  gains carry the 255 full-scale factor
  c.cy = (int)std::lround(Q * 255.0 / yscale);
  c.crv = (int)std::lround(Q * 255.0 * 2.0 * (1.0 - kr) / cscale);
  c.cbu = (int)std::lround(Q * 255.0 * 2.0 * (1.0 - kb) / cscale);
  c.cgu = (int)std::lround(Q * 255.0 * 2.0 * kb * (1.0 - kb) / kg / cscale);
  c.cgv = (int)std::lround(Q * 255.0 * 2.0 * kr * (1.0 - kr) / kg / cscale);
  c.fyr = (int)std::lround(Q * kr * yscale / 255.0);
  c.fyg = (int)std::lround(Q * kg * yscale / 255.0);
  c.fyb = (int)std::lround(Q * kb * yscale / 255.0);
  c.y_gain = (float)(c.cy / Q);
  return c;
}

inline int clamp255(int v) { return v < 0 ? 0 : (v > 255 ? 255 : v); }

// sRGB transfer functions (IEC 61966-2-1).
inline float srgbToLin(float c) {
  return c <= 0.04045f ? c * (1.0f / 12.92f) : std::pow((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}
inline float linToSrgb(float c) {
  if (c <= 0.0f) return 0.0f;
  if (c >= 1.0f) return 1.0f;
  return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

}  // namespace hie
