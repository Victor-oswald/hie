// Path A: analytic "AI brain". Deterministic statistics on the 256x256 thumbnail -> Params.
// A learned model (IA-3DLUT, Zero-DCE++, MobileNetV3 regressor) plugs in through hie::Predictor
// and refines the result; this analytic predictor is the always-available baseline.
#pragma once
#include <vector>

#include "common.hpp"

namespace hie {

// Preallocated working memory (analysis must not allocate on the real-time path).
struct AnalysisScratch {
  std::vector<float> lin;
  AnalysisScratch() : lin((size_t)3 * (kThumbSize / 2) * (kThumbSize / 2)) {}
};

// out is initialised from cfg.manual; automatic parameters are then overwritten.
// Allocation-free; safe on any thread.
void analyzeThumbnail(const Thumbnail& t, const Config& cfg, Params& out, AnalysisScratch& sc);

// Robust noise sigma (luma levels) from a kCropSize^2 native-resolution crop.
float estimateNoiseSigma(const uint8_t* crop, int size);

}  // namespace hie
