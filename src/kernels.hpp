// Path-B pixel kernels. All are allocation-free and operate on caller-provided buffers.
// Hot loops are plain C++17 written for auto-vectorisation (AVX2/SSE4.1 via function
// multiversioning on x86, NEON on ARM).
#pragma once
#include <cstdint>

#include "common.hpp"

namespace hie {
namespace k {

// --- ingest -----------------------------------------------------------------------------------
void yuyvExtractY(const uint8_t* src, int sstride, uint8_t* dst, int dstride, int w, int y0,
                         int y1);
void rgbToLuma(const uint8_t* src, int sstride, uint8_t* dst, int dstride, int w, int y0,
                      int y1, const YuvCoeffs& c);
void gatherStrided(const uint8_t* src, int step, int n, uint8_t* dst);

// --- luma ---------------------------------------------------------------------------------------
// Edge-preserving 3x3 filter: weight = spatial * max(0, T - |dI|)   (T >= 1).
void denoiseRows(const uint8_t* src, int ss, uint8_t* dst, int ds, int w, int h, int y0,
                        int y1, int T);

// CLAHE tile mapping, 256 entries in 8.8 fixed point, output range [lo,hi].
void claheTileLut(const uint8_t* y, int stride, int x0, int y0, int x1, int y1, int hstride,
                  float clip, float strength, int lo, int hi, uint16_t* lut);
// out = (a*(256-fy) + b*fy + 128) >> 8
void claheVBlend(const uint16_t* a, const uint16_t* b, int fy, uint16_t* out, int n);

struct ClaheSeg { int x0, x1, t0, t1; };   // pixel run sharing the same pair of tile columns
// Applies horizontal interpolation of the vertically-blended per-tile LUTs.
void claheApplyRow(const uint8_t* yin, uint8_t* yout, const ClaheSeg* segs, int nseg,
                   const uint16_t* fx, const uint16_t* rowLut);

// --- chroma / colour -----------------------------------------------------------------------------
void vblend31(const uint8_t* a, const uint8_t* b, uint8_t* out, int n);
void hupLinear(const uint8_t* c, int cw, uint8_t* out, int w);
void hupNearest(const uint8_t* c, uint8_t* out, int w);
void yuvToRgb(const uint8_t* Y, const uint8_t* U, const uint8_t* V, int w, const YuvCoeffs& c,
                     uint8_t* R, uint8_t* G, uint8_t* B);
// RGB24 input path: add the luma change to the untouched source colour (exact when dY == 0).
void rgbApplyLumaDelta(const uint8_t* rgb, const uint8_t* yEnh, const uint8_t* yOrig, int w,
                              int cy, uint8_t* R, uint8_t* G, uint8_t* B);

}  // namespace k
}  // namespace hie
