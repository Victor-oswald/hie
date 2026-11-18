// Fused colour transform (white balance + exposure + tone + saturation + look LUTs) baked into a
// 3D LUT, plus the fixed-point tetrahedral / trilinear evaluators used on every pixel.
#pragma once
#include <cstdint>
#include <vector>

#include "common.hpp"
#include "threadpool.hpp"

namespace hie {

// Exact (float) transform the LUT approximates. in/out are display-referred sRGB in [0,1].
// `bank` may be null; external LUTs are blended with Params::lut_weights.
void exactTransform(const Params& p, const std::vector<Lut3D>* bank, const float in[3], float out[3]);

// Tone curve shared by the exact transform and the factored pipeline (y in [0,1]).
float toneCurveShared(float y, float lift, float c);

// Float trilinear sample of an external LUT.
void sampleLut3D(const Lut3D& l, const float in[3], float out[3]);

struct LutRT {
  int n = 0;
  float shaper = 1.0f;                 // node placement exponent (S(x) = x^shaper)
  bool identity = true;
  std::vector<uint16_t> nodes;         // n^3 * 4 (R,G,B,pad), 0..65280
  // per input value tables (shared by all channels)
  uint8_t idx[256];
  uint16_t frac[256];                  // 0..4096
  std::vector<uint32_t> offR, offG, offB;  // element offsets (256 each)
  std::vector<float> axis;             // node coordinates (preallocated)
  uint32_t sR = 0, sG = 0, sB = 4;     // element strides

  void init(int size, float shaper_gamma);   // allocates once
  // Bakes p (and the bank) into the nodes. Parallel over red slabs.
  void build(const Params& p, const std::vector<Lut3D>* bank, ThreadPool& pool);
};

// Test hook: force the portable scalar LUT kernel even when SIMD is available.
void setForceScalarLut(bool on);

// Row appliers: planar R,G,B (8-bit) -> packed output.
void lutApplyRow(const LutRT& lut, bool tetra, const uint8_t* r, const uint8_t* g, const uint8_t* b,
                 int w, OutFormat fmt, uint8_t* dst);
void packRow(const uint8_t* r, const uint8_t* g, const uint8_t* b, int w, OutFormat fmt,
             uint8_t* dst);

}  // namespace hie
