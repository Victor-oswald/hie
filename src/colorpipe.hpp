// Factored colour pipeline: the analytic transform (white balance, exposure, tone, saturation)
// evaluated EXACTLY through small tables instead of a sampled 3D LUT:
//   d_c  = T_c[in_c]                 (per-channel: sRGB decode -> WB*exposure gain -> sRGB encode)
//   y    = luma(d)                    (Q16)
//   out  = clamp(k1[y] * d + k0[y])   (tone curve as a luma ratio + saturation)
// 4 lookups / pixel, no interpolation error, gather-friendly. Matches exactTransform() (no bank).
#pragma once
#include <cstdint>

#include "common.hpp"

namespace hie {

struct FactoredRT {
  bool identity = true;
  int32_t T[3][256];       // Q16 display value after gains
  uint32_t K[4096];        // low 16 bits: k1 (Q12), high 16 bits: k0 (int16, Q15)
  void build(const Params& p);
};

// In place on planar 8-bit rows.
void factoredRow(const FactoredRT& f, uint8_t* r, uint8_t* g, uint8_t* b, int w);

}  // namespace hie
