// Hybrid Image Enhancement engine (HIE) - public C++17 API.
//
//   Path A (async, every N frames) : 256x256 thumbnail -> predictor -> target Params -> EMA
//   Path B (every frame, SIMD/MT)  : Y-only denoise + CLAHE -> chroma upsample -> YUV->RGB
//                                     -> fused 3D LUT (white balance, exposure, tone, saturation)
//
// Thread-safety: one Enhancer instance must be driven from a single thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#define HIE_VERSION_STRING "1.0.0"

namespace hie {

enum class InFormat : uint8_t {
  I420,   // planar Y, U, V (4:2:0)           data[0..2]
  NV12,   // planar Y + interleaved UV (4:2:0) data[0..1]
  YUYV,   // packed Y0 U Y1 V (4:2:2, "YUY2") data[0]
  RGB24   // packed R G B (display-referred sRGB)  data[0]
};
enum class OutFormat : uint8_t { RGB24, BGR24, RGBA32, BGRA32 };
enum class Matrix : uint8_t { BT601, BT709 };
enum class Range : uint8_t { Limited, Full };
enum class ChromaUpsample : uint8_t { Nearest, Linear };

constexpr int kMaxLuts = 4;       // external base LUTs that can be blended
constexpr int kThumbSize = 256;   // Path-A thumbnail edge
constexpr int kCropSize = 128;    // native-resolution luma crop (noise estimate)

// Everything Path A predicts / Path B consumes. A default-constructed Params is "neutral".
struct Params {
  float wb_r = 1.f, wb_b = 1.f;  // linear-light gains relative to green (luminance-normalised)
  float exposure_ev = 0.f;       // stops, applied in linear light
  float contrast = 1.f;          // 1 = none; S-curve about mid grey (0.7 .. 1.5)
  float saturation = 1.f;        // 1 = none
  float shadow_lift = 0.f;       // 0 .. 0.8
  float clahe_strength = 0.f;    // 0 .. 1 blend between identity and CLAHE mapping
  float clahe_clip = 2.5f;       // CLAHE clip limit (multiples of the uniform histogram)
  float denoise_t = 0.f;         // bilateral range threshold in luma levels (0 = off)
  float lut_weights[kMaxLuts] = {0, 0, 0, 0};  // blend weights of external base LUTs
};

// result = a*(1-w) + b*w; white-balance gains are interpolated geometrically.
Params blendParams(const Params& a, const Params& b, float w);
// True when the colour LUT built from p would be the identity (LUT stage can be skipped).
bool lutIsNeutral(const Params& p);

// What a Predictor sees (Path A input). All data is display-referred, after the YUV matrix,
// before any enhancement.
struct Thumbnail {
  const uint8_t* rgb;        // kThumbSize x kThumbSize x 3
  const uint8_t* luma_crop;  // kCropSize x kCropSize native-resolution luma
};

// Plug-in point for a learned model (IA-3DLUT / MobileNetV3 regressor via NCNN / ONNX / LiteRT).
// `inout` arrives pre-filled with the built-in analytic estimate; overwrite what you predict.
// Called on the analysis thread (or inline in sync mode). Must not touch the Enhancer.
class Predictor {
 public:
  virtual ~Predictor() = default;
  virtual void refine(const Thumbnail& thumb, Params& inout) = 0;
};

// External 3D LUT ("look"), .cube compatible. data = n^3 RGB float triplets, red fastest.
struct Lut3D {
  int size = 0;
  std::vector<float> data;
  static Lut3D identity(int n);
  static bool loadCube(const std::string& path, Lut3D& out, std::string* err = nullptr);
};

struct Config {
  int width = 0, height = 0;
  InFormat in_format = InFormat::I420;
  OutFormat out_format = OutFormat::RGB24;
  Matrix matrix = Matrix::BT601;           // webcams / SD: BT601, HD: BT709
  Range range = Range::Limited;
  ChromaUpsample chroma_upsample = ChromaUpsample::Linear;

  int threads = 0;                         // 0 = auto (<= 8)

  // Path A scheduling
  int analysis_interval = 12;              // frames between analyses (10-15 per spec)
  bool async_analysis = true;              // false: analyse inline (deterministic, for stills/tests)
  bool sync_first_frame = true;            // first frame is analysed inline so it is already enhanced
  float ema_old = 0.85f;                   // W = ema_old*W_prev + (1-ema_old)*W_new
  float frame_smoothing = 0.15f;           // per-frame easing toward the EMA state (1 = off)

  // Stage switches
  bool enable_denoise = true;
  bool enable_clahe = true;
  bool enable_lut = true;

  // Which parameters Path A may drive; others use `manual`.
  bool auto_wb = true, auto_exposure = true, auto_contrast = true, auto_saturation = true,
       auto_shadows = true, auto_clahe = true, auto_denoise = true;
  Params manual;                           // value used for every non-automatic parameter

  // Analytic predictor tuning
  float wb_strength = 0.75f;               // 0..1 how much of the estimated cast is removed
  float exposure_strength = 0.6f;
  float target_median = 0.46f;             // display-referred median luma goal
  float clahe_strength_max = 0.85f;
  float clahe_clip_base = 2.5f;

  // CLAHE
  int clahe_tiles_x = 8, clahe_tiles_y = 8;
  int clahe_hist_stride = 2;               // histogram sub-sampling (1 = every pixel)
  float clahe_temporal = 0.3f;             // per-frame blend of new tile LUTs (anti-flicker)

  // 3D LUT
  int lut_size = 17;                       // 17 (L1/L2 resident) or 33 (higher accuracy)
  float shaper_gamma = 1.00f;              // node placement exponent (see docs/ACCURACY.md)
  bool tetrahedral = true;                 // false = trilinear
  // Analytic colour params are normally evaluated exactly via factored tables (faster, no
  // interpolation error). External .cube looks always use the 3D LUT. true = force the 3D LUT.
  bool prefer_3d_lut = false;
};

struct FrameIn {
  const uint8_t* data[3] = {nullptr, nullptr, nullptr};
  int stride[3] = {0, 0, 0};
  // Tightly packed frame in one allocation (I420: Y,U,V; NV12: Y,UV; YUYV/RGB24: packed).
  static FrameIn contiguous(InFormat f, int w, int h, const uint8_t* p);
};

struct Stats {
  uint64_t frames = 0;
  uint64_t analyses = 0;
  uint64_t lut_rebuilds = 0;
  bool lut_identity = true;
  double last_frame_ms = 0;
  // last-frame stage breakdown (ms): ingest+analysis scheduling, denoise, CLAHE tiles, fused output
  double ms_prep = 0, ms_denoise = 0, ms_clahe = 0, ms_output = 0;
};

class Enhancer {
 public:
  explicit Enhancer(const Config& cfg);   // allocates everything once
  ~Enhancer();
  Enhancer(const Enhancer&) = delete;
  Enhancer& operator=(const Enhancer&) = delete;

  bool valid() const;
  const std::string& error() const;       // set when !valid()

  // Enhances one frame. dst must hold height rows of dst_stride bytes. No heap allocation.
  bool process(const FrameIn& in, uint8_t* dst, int dst_stride);

  void reset();                            // forget temporal state
  bool addLut(const Lut3D& lut);           // register external base LUT (index = order added)
  void setPredictor(std::shared_ptr<Predictor> p);
  void setManual(const Params& p);         // changes values of non-automatic parameters
  Params currentParams() const;            // what the last frame was rendered with
  Params targetParams() const;             // EMA state
  Stats stats() const;
  const Config& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Pre-allocated output ring (spec: "pre-allocated reusable ring buffers").
class FrameRing {
 public:
  FrameRing(int slots, size_t bytes_per_frame);
  uint8_t* acquire();                      // next slot, round-robin
  size_t frameBytes() const { return bytes_; }
 private:
  std::vector<uint8_t> mem_;
  size_t bytes_;
  int slots_, next_ = 0;
};

}  // namespace hie
