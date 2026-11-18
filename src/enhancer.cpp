#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

#include "analysis.hpp"
#include "common.hpp"
#include "kernels.hpp"
#include "colorpipe.hpp"
#include "lut3d.hpp"
#include "threadpool.hpp"

namespace hie {

// ---------------------------------------------------------------------------------------------
// FrameIn / FrameRing
// ---------------------------------------------------------------------------------------------
FrameIn FrameIn::contiguous(InFormat f, int w, int h, const uint8_t* p) {
  FrameIn r;
  const int cw = (w + 1) / 2, ch = (h + 1) / 2;
  switch (f) {
    case InFormat::I420:
      r.data[0] = p; r.stride[0] = w;
      r.data[1] = p + (size_t)w * h; r.stride[1] = cw;
      r.data[2] = r.data[1] + (size_t)cw * ch; r.stride[2] = cw;
      break;
    case InFormat::NV12:
      r.data[0] = p; r.stride[0] = w;
      r.data[1] = p + (size_t)w * h; r.stride[1] = cw * 2;
      break;
    case InFormat::YUYV:
      r.data[0] = p; r.stride[0] = cw * 4;
      break;
    case InFormat::RGB24:
      r.data[0] = p; r.stride[0] = w * 3;
      break;
  }
  return r;
}

FrameRing::FrameRing(int slots, size_t bytes) : mem_((size_t)(slots < 1 ? 1 : slots) * bytes), bytes_(bytes), slots_(slots < 1 ? 1 : slots) {}
uint8_t* FrameRing::acquire() {
  uint8_t* p = mem_.data() + (size_t)next_ * bytes_;
  next_ = (next_ + 1) % slots_;
  return p;
}

namespace {

bool lutFieldsDiffer(const Params& a, const Params& b) {
  if (std::fabs(a.wb_r - b.wb_r) > 2e-4f || std::fabs(a.wb_b - b.wb_b) > 2e-4f) return true;
  if (std::fabs(a.exposure_ev - b.exposure_ev) > 2e-3f) return true;
  if (std::fabs(a.contrast - b.contrast) > 1e-3f || std::fabs(a.saturation - b.saturation) > 1e-3f) return true;
  if (std::fabs(a.shadow_lift - b.shadow_lift) > 1e-3f) return true;
  for (int i = 0; i < kMaxLuts; ++i)
    if (std::fabs(a.lut_weights[i] - b.lut_weights[i]) > 1e-3f) return true;
  return false;
}

bool paramsClose(const Params& a, const Params& b) {
  if (lutFieldsDiffer(a, b)) return false;
  return std::fabs(a.clahe_strength - b.clahe_strength) < 1e-3f &&
         std::fabs(a.clahe_clip - b.clahe_clip) < 1e-3f && std::fabs(a.denoise_t - b.denoise_t) < 1e-2f;
}

struct Scratch {
  std::vector<uint8_t> yrow, urow, vrow, tA, tB, tV, R, G, B;
  std::vector<uint16_t> rowLut;
};

}  // namespace

// ---------------------------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------------------------
struct Enhancer::Impl {
  Config cfg;
  bool ok = true;
  std::string err;
  YuvCoeffs yc;
  int w = 0, h = 0, cw = 0, ch = 0;
  int csh = 1;  // vertical chroma shift (1 = 4:2:0, 0 = 4:2:2)
  bool rgbIn = false;

  std::unique_ptr<ThreadPool> pool;
  std::vector<Scratch> scr;

  // planes
  std::vector<uint8_t> yIn, yDn;
  const uint8_t* ysrc = nullptr; int ystride = 0;
  const uint8_t* yd = nullptr; int ydStride = 0;
  const uint8_t* uBase = nullptr; const uint8_t* vBase = nullptr;
  int uStride = 0, vStride = 0, cstep = 1;
  const uint8_t* rgbSrc = nullptr; int rgbStride = 0;

  // CLAHE
  int ntx = 1, nty = 1;
  std::vector<uint16_t> tileState;       // ntx*nty*256
  bool tileReset = true;
  std::vector<k::ClaheSeg> segs;
  std::vector<uint16_t> fxTab;           // per x, 0..256
  std::vector<uint8_t> rowJ0;            // per y
  std::vector<uint16_t> rowFy;           // per y

  // LUT
  LutRT lut;
  FactoredRT fact;
  bool mode3d = false, builtMode3d = false;
  std::vector<Lut3D> bank;
  Params built;
  bool lutBuilt = false;

  // temporal state
  Params cur, ema;
  bool haveState = false;
  bool first = true;
  uint64_t frameNo = 0;
  Stats st;

  // Path A
  std::vector<uint8_t> thumb, crop;
  int thX[kThumbSize][2], thY[kThumbSize][2];
  std::thread worker;
  std::mutex wm, resMx, predMx, cfgMx;
  std::condition_variable wcv;
  bool hasJob = false, stopWorker = false;
  std::atomic<bool> busy{false}, resultReady{false};
  Params result;
  std::shared_ptr<Predictor> pred;
  AnalysisScratch ascratch;
  Config liveCfg;                         // snapshot used by analysis (manual may change)
  bool forceAnalysis = false;

  explicit Impl(const Config& c) : cfg(c) {
    if (cfg.width < 2 || cfg.height < 2 || cfg.width > 16384 || cfg.height > 16384) {
      ok = false; err = "invalid frame size"; return;
    }
    if (cfg.in_format == InFormat::YUYV && (cfg.width & 1)) {
      ok = false; err = "YUYV requires an even width"; return;
    }
    if (cfg.lut_size < 5 || cfg.lut_size > 65) { ok = false; err = "lut_size must be in [5,65]"; return; }
    if (cfg.shaper_gamma < 0.3f || cfg.shaper_gamma > 1.5f) { ok = false; err = "bad shaper_gamma"; return; }
    if (cfg.clahe_hist_stride < 1) cfg.clahe_hist_stride = 1;
    if (cfg.ema_old < 0.f) cfg.ema_old = 0.f;
    if (cfg.ema_old > 1.f) cfg.ema_old = 1.f;
    if (cfg.frame_smoothing <= 0.f || cfg.frame_smoothing > 1.f) cfg.frame_smoothing = 1.f;

    w = cfg.width; h = cfg.height;
    cw = (w + 1) / 2; ch = (h + 1) / 2;
    csh = (cfg.in_format == InFormat::YUYV) ? 0 : 1;
    if (csh == 0) ch = h;
    rgbIn = cfg.in_format == InFormat::RGB24;
    yc = makeCoeffs(cfg.matrix, cfg.range);

    int nt = cfg.threads;
    if (nt <= 0) {
      nt = (int)std::thread::hardware_concurrency();
      if (nt <= 0) nt = 1;
      if (nt > 8) nt = 8;
    }
    pool.reset(new ThreadPool(nt));
    scr.resize(nt);
    ntx = std::max(1, std::min(cfg.clahe_tiles_x, w / 16));
    nty = std::max(1, std::min(cfg.clahe_tiles_y, h / 16));
    for (auto& s : scr) {
      s.yrow.assign(w + 16, 0); s.urow.assign(w + 16, 0); s.vrow.assign(w + 16, 0);
      s.tA.assign(cw + 16, 0); s.tB.assign(cw + 16, 0); s.tV.assign(cw + 16, 0);
      s.R.assign(w + 16, 0); s.G.assign(w + 16, 0); s.B.assign(w + 16, 0);
      s.rowLut.assign((size_t)ntx * 256, 0);
    }
    if (cfg.in_format == InFormat::YUYV || rgbIn) yIn.assign((size_t)w * h, 0);
    yDn.assign((size_t)w * h, 0);

    // CLAHE geometry
    tileState.assign((size_t)ntx * nty * 256, 0);
    fxTab.assign(w, 0);
    rowJ0.assign(h, 0);
    rowFy.assign(h, 0);
    {
      auto pos1 = [](int x, int n, float tile) {   // interpolation position in tile units
        float p = ((float)x + 0.5f) / tile - 0.5f;
        return std::min(std::max(p, 0.f), (float)(n - 1));
      };
      const float tw = (float)w / ntx, th = (float)h / nty;
      int runStart = 0, runT0 = -1;
      for (int x = 0; x < w; ++x) {
        float p = pos1(x, ntx, tw);
        int i0 = ntx > 1 ? std::min((int)p, ntx - 2) : 0;
        fxTab[x] = ntx > 1 ? (uint16_t)std::lround((p - i0) * 256.f) : 0;
        if (i0 != runT0) {
          if (runT0 >= 0) segs.push_back({runStart, x, runT0, std::min(runT0 + 1, ntx - 1)});
          runStart = x; runT0 = i0;
        }
      }
      segs.push_back({runStart, w, runT0, std::min(runT0 + 1, ntx - 1)});
      for (int y = 0; y < h; ++y) {
        float p = pos1(y, nty, th);
        int j0 = nty > 1 ? std::min((int)p, nty - 2) : 0;
        rowJ0[y] = (uint8_t)j0;
        rowFy[y] = nty > 1 ? (uint16_t)std::lround((p - j0) * 256.f) : 0;
      }
    }

    lut.init(cfg.lut_size, cfg.shaper_gamma);
    cur = ema = built = cfg.manual;

    thumb.assign((size_t)kThumbSize * kThumbSize * 3, 0);
    crop.assign((size_t)kCropSize * kCropSize, 0);
    for (int i = 0; i < kThumbSize; ++i)
      for (int k2 = 0; k2 < 2; ++k2) {
        thX[i][k2] = std::min(w - 1, ((4 * i + 1 + 2 * k2) * w) / (4 * kThumbSize));
        thY[i][k2] = std::min(h - 1, ((4 * i + 1 + 2 * k2) * h) / (4 * kThumbSize));
      }
    liveCfg = cfg;
    if (cfg.async_analysis) worker = std::thread([this] { workerLoop(); });
  }

  ~Impl() {
    if (worker.joinable()) {
      { std::lock_guard<std::mutex> lk(wm); stopWorker = true; hasJob = true; }
      wcv.notify_one();
      worker.join();
    }
  }

  // ---- Path A ----------------------------------------------------------------------------------
  void runAnalysis(Params& out) {
    Config c;
    std::shared_ptr<Predictor> p;
    { std::lock_guard<std::mutex> lk(cfgMx); c = liveCfg; }
    { std::lock_guard<std::mutex> lk(predMx); p = pred; }
    Thumbnail t{thumb.data(), crop.data()};
    analyzeThumbnail(t, c, out, ascratch);
    if (p) p->refine(t, out);
  }

  void workerLoop() {
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(wm);
        wcv.wait(lk, [this] { return hasJob; });
        if (stopWorker) return;
        hasJob = false;
      }
      Params r;
      runAnalysis(r);
      { std::lock_guard<std::mutex> lk(resMx); result = r; }
      resultReady.store(true, std::memory_order_release);
      busy.store(false, std::memory_order_release);
    }
  }

  inline void yuvPix(int y, int u, int v, uint8_t* o) const {
    int yy = (y - yc.ylo) * yc.cy + 8192;
    u -= 128; v -= 128;
    o[0] = (uint8_t)clamp255((yy + yc.crv * v) >> 14);
    o[1] = (uint8_t)clamp255((yy - yc.cgu * u - yc.cgv * v) >> 14);
    o[2] = (uint8_t)clamp255((yy + yc.cbu * u) >> 14);
  }

  void buildThumbnail() {
    pool->run(kThumbSize / 8, [&](int band, int) {
      for (int ty = band * 8; ty < band * 8 + 8; ++ty) {
        uint8_t* o = thumb.data() + (size_t)ty * kThumbSize * 3;
        for (int tx = 0; tx < kThumbSize; ++tx, o += 3) {
          if (rgbIn) {
            int r = 0, g = 0, b = 0;
            for (int ky = 0; ky < 2; ++ky)
              for (int kx = 0; kx < 2; ++kx) {
                const uint8_t* s = rgbSrc + (size_t)thY[ty][ky] * rgbStride + (size_t)thX[tx][kx] * 3;
                r += s[0]; g += s[1]; b += s[2];
              }
            o[0] = (uint8_t)((r + 2) >> 2); o[1] = (uint8_t)((g + 2) >> 2); o[2] = (uint8_t)((b + 2) >> 2);
          } else {
            int ys = 0, us = 0, vs = 0;
            for (int ky = 0; ky < 2; ++ky) {
              const int sy = thY[ty][ky];
              const uint8_t* yr = ysrc + (size_t)sy * ystride;
              const size_t cr = (size_t)(sy >> csh);
              for (int kx = 0; kx < 2; ++kx) {
                const int sx = thX[tx][kx];
                ys += yr[sx];
                us += uBase[cr * uStride + (size_t)(sx >> 1) * cstep];
                vs += vBase[cr * vStride + (size_t)(sx >> 1) * cstep];
              }
            }
            yuvPix((ys + 2) >> 2, (us + 2) >> 2, (vs + 2) >> 2, o);
          }
        }
      }
    });
    // native-resolution luma crop (noise statistics must not be low-passed by the downscale)
    const int cx0 = std::max((w - kCropSize) / 2, 0), cy0 = std::max((h - kCropSize) / 2, 0);
    for (int y = 0; y < kCropSize; ++y) {
      const uint8_t* s = ysrc + (size_t)std::min(cy0 + y, h - 1) * ystride;
      uint8_t* d = crop.data() + (size_t)y * kCropSize;
      if (w >= kCropSize) std::memcpy(d, s + cx0, kCropSize);
      else for (int x = 0; x < kCropSize; ++x) d[x] = s[std::min(x, w - 1)];
    }
  }

  void applyResult(const Params& r, bool warm) {
    if (!haveState || warm) {
      ema = r;
      cur = r;
      haveState = true;
    } else {
      ema = blendParams(ema, r, 1.0f - cfg.ema_old);   // W = 0.85*W_prev + 0.15*W_new
    }
    ++st.analyses;
  }

  void schedulePathA() {
    const bool interval_hit = cfg.analysis_interval > 0 && (frameNo % (uint64_t)cfg.analysis_interval) == 0;
    const bool want = first || forceAnalysis || interval_hit;
    // 1. collect finished background analysis
    if (resultReady.exchange(false, std::memory_order_acq_rel)) {
      Params r;
      { std::lock_guard<std::mutex> lk(resMx); r = result; }
      applyResult(r, false);
    }
    if (!want) return;
    const bool inline_now = !cfg.async_analysis || (first && cfg.sync_first_frame) || (forceAnalysis && !busy.load());
    if (inline_now) {
      if (busy.load(std::memory_order_acquire)) return;   // never race the worker on shared buffers
      buildThumbnail();
      Params r;
      runAnalysis(r);
      applyResult(r, first);
      forceAnalysis = false;
    } else if (!busy.load(std::memory_order_acquire)) {
      buildThumbnail();
      busy.store(true, std::memory_order_release);
      { std::lock_guard<std::mutex> lk(wm); hasJob = true; }
      wcv.notify_one();
    }
  }

  // ---- per frame ---------------------------------------------------------------------------------
  void updateParams() {
    if (cfg.frame_smoothing >= 1.f) cur = ema;
    else if (!paramsClose(cur, ema)) {
      cur = blendParams(cur, ema, cfg.frame_smoothing);
      if (paramsClose(cur, ema)) cur = ema;
    }
    if (!cfg.enable_lut) { lut.identity = true; fact.identity = true; mode3d = false; return; }
    bool bankOn = false;
    for (size_t i = 0; i < bank.size() && i < (size_t)kMaxLuts; ++i)
      if (cur.lut_weights[i] > 1e-4f) bankOn = true;
    mode3d = cfg.prefer_3d_lut || bankOn;
    if (!lutBuilt || mode3d != builtMode3d || lutFieldsDiffer(cur, built)) {
      if (mode3d) lut.build(cur, bank.empty() ? nullptr : &bank, *pool);
      else fact.build(cur);
      built = cur;
      builtMode3d = mode3d;
      lutBuilt = true;
      ++st.lut_rebuilds;
    }
  }

  const uint8_t* chromaRow(const uint8_t* base, int stride, int row, uint8_t* tmp) const {
    const uint8_t* s = base + (size_t)row * stride;
    if (cstep == 1) return s;
    k::gatherStrided(s, cstep, cw, tmp);
    return tmp;
  }

  void chromaFor(int y, Scratch& sc) {
    const int cy = y >> csh;
    const bool lin = cfg.chroma_upsample == ChromaUpsample::Linear;
    for (int pl = 0; pl < 2; ++pl) {
      const uint8_t* base = pl == 0 ? uBase : vBase;
      const int stride = pl == 0 ? uStride : vStride;
      uint8_t* dst = pl == 0 ? sc.urow.data() : sc.vrow.data();
      const uint8_t* a = chromaRow(base, stride, cy, sc.tA.data());
      if (lin && csh == 1) {
        int nb = (y & 1) ? std::min(cy + 1, ch - 1) : std::max(cy - 1, 0);
        const uint8_t* b = chromaRow(base, stride, nb, sc.tB.data());
        k::vblend31(a, b, sc.tV.data(), cw);
        a = sc.tV.data();
      }
      if (lin) k::hupLinear(a, cw, dst, w);
      else k::hupNearest(a, dst, w);
    }
  }

  bool process(const FrameIn& in, uint8_t* dst, int dstride) {
    const auto t0 = std::chrono::steady_clock::now();
    if (!ok || !dst || dstride < w * ((cfg.out_format == OutFormat::RGB24 || cfg.out_format == OutFormat::BGR24) ? 3 : 4))
      return false;
    switch (cfg.in_format) {
      case InFormat::I420:
        if (!in.data[0] || !in.data[1] || !in.data[2]) return false;
        ysrc = in.data[0]; ystride = in.stride[0];
        uBase = in.data[1]; uStride = in.stride[1]; vBase = in.data[2]; vStride = in.stride[2];
        cstep = 1;
        break;
      case InFormat::NV12:
        if (!in.data[0] || !in.data[1]) return false;
        ysrc = in.data[0]; ystride = in.stride[0];
        uBase = in.data[1]; vBase = in.data[1] + 1; uStride = vStride = in.stride[1];
        cstep = 2;
        break;
      case InFormat::YUYV:
        if (!in.data[0]) return false;
        uBase = in.data[0] + 1; vBase = in.data[0] + 3; uStride = vStride = in.stride[0];
        cstep = 4;
        pool->run((h + 31) / 32, [&](int b, int) {
          k::yuyvExtractY(in.data[0], in.stride[0], yIn.data(), w, w, b * 32, std::min(h, b * 32 + 32));
        });
        ysrc = yIn.data(); ystride = w;
        break;
      case InFormat::RGB24:
        if (!in.data[0]) return false;
        rgbSrc = in.data[0]; rgbStride = in.stride[0];
        pool->run((h + 31) / 32, [&](int b, int) {
          k::rgbToLuma(in.data[0], in.stride[0], yIn.data(), w, w, b * 32, std::min(h, b * 32 + 32), yc);
        });
        ysrc = yIn.data(); ystride = w;
        break;
    }

    schedulePathA();
    updateParams();
    const auto tA = std::chrono::steady_clock::now();

    // ---- Path B pass 1: edge-preserving luma denoise --------------------------------------------
    const int T = (int)std::lround(cur.denoise_t);
    const bool doDenoise = cfg.enable_denoise && T >= 2;
    if (doDenoise) {
      const uint8_t* s = ysrc; const int ss = ystride;
      pool->run((h + 15) / 16, [&](int b, int) {
        k::denoiseRows(s, ss, yDn.data(), w, w, h, b * 16, std::min(h, b * 16 + 16), T);
      });
      yd = yDn.data(); ydStride = w;
    } else {
      yd = ysrc; ydStride = ystride;
    }

    const auto tB = std::chrono::steady_clock::now();
    // ---- pass 2: CLAHE tile mappings (temporally smoothed) -----------------------------------------
    const bool doClahe = cfg.enable_clahe && cur.clahe_strength > 0.02f;
    if (doClahe) {
      const float clip = cur.clahe_clip, strength = std::min(cur.clahe_strength, 1.f);
      int a = (int)std::lround(cfg.clahe_temporal * 256.f);
      a = std::min(std::max(a, 1), 256);
      const bool reset = tileReset || a >= 256;
      pool->run(ntx * nty, [&](int t, int) {
        const int tx = t % ntx, ty = t / ntx;
        uint16_t fresh[256];
        k::claheTileLut(yd, ydStride, tx * w / ntx, ty * h / nty, (tx + 1) * w / ntx,
                        (ty + 1) * h / nty, cfg.clahe_hist_stride, clip, strength, yc.ymin,
                        yc.ymax, fresh);
        uint16_t* S = tileState.data() + (size_t)t * 256;
        if (reset) std::memcpy(S, fresh, sizeof(fresh));
        else for (int i = 0; i < 256; ++i) S[i] = (uint16_t)(((int)S[i] * (256 - a) + (int)fresh[i] * a + 128) >> 8);
      });
      tileReset = false;
    } else {
      tileReset = true;
    }

    const auto tC = std::chrono::steady_clock::now();
    // ---- pass 3: fused  CLAHE-apply + chroma upsample + YUV->RGB + 3D LUT + pack ------------------
    const bool colorIdentity = !cfg.enable_lut || (mode3d ? lut.identity : fact.identity);
    const bool useLut = !colorIdentity && mode3d;
    const bool useFact = !colorIdentity && !mode3d;
    const bool tetra = cfg.tetrahedral;
    const int bandRows = 8;
    pool->run((h + bandRows - 1) / bandRows, [&](int band, int tid) {
      Scratch& sc = scr[tid];
      const int r1 = std::min(h, (band + 1) * bandRows);
      for (int y = band * bandRows; y < r1; ++y) {
        const uint8_t* yrow = yd + (size_t)y * ydStride;
        if (doClahe) {
          const int j0 = rowJ0[y], j1 = std::min(j0 + 1, nty - 1), fy = rowFy[y];
          for (int t = 0; t < ntx; ++t)
            k::claheVBlend(tileState.data() + ((size_t)j0 * ntx + t) * 256,
                           tileState.data() + ((size_t)j1 * ntx + t) * 256, fy,
                           sc.rowLut.data() + (size_t)t * 256, 256);
          k::claheApplyRow(yrow, sc.yrow.data(), segs.data(), (int)segs.size(), fxTab.data(),
                           sc.rowLut.data());
          yrow = sc.yrow.data();
        }
        if (rgbIn) {
          k::rgbApplyLumaDelta(rgbSrc + (size_t)y * rgbStride, yrow, ysrc + (size_t)y * ystride, w,
                               yc.cy, sc.R.data(), sc.G.data(), sc.B.data());
        } else {
          chromaFor(y, sc);
          k::yuvToRgb(yrow, sc.urow.data(), sc.vrow.data(), w, yc, sc.R.data(), sc.G.data(), sc.B.data());
        }
        uint8_t* o = dst + (size_t)y * dstride;
        if (useLut) lutApplyRow(lut, tetra, sc.R.data(), sc.G.data(), sc.B.data(), w, cfg.out_format, o);
        else if (useFact) {
          factoredRow(fact, sc.R.data(), sc.G.data(), sc.B.data(), w);
          packRow(sc.R.data(), sc.G.data(), sc.B.data(), w, cfg.out_format, o);
        } else packRow(sc.R.data(), sc.G.data(), sc.B.data(), w, cfg.out_format, o);
      }
    });

    const auto tD = std::chrono::steady_clock::now();
    auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    st.ms_prep = ms(t0, tA); st.ms_denoise = ms(tA, tB); st.ms_clahe = ms(tB, tC); st.ms_output = ms(tC, tD);
    first = false;
    ++frameNo;
    ++st.frames;
    st.lut_identity = colorIdentity;
    st.last_frame_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
  }
};

// ---------------------------------------------------------------------------------------------
// Enhancer facade
// ---------------------------------------------------------------------------------------------
Enhancer::Enhancer(const Config& cfg) : impl_(new Impl(cfg)) {}
Enhancer::~Enhancer() = default;
bool Enhancer::valid() const { return impl_->ok; }
const std::string& Enhancer::error() const { return impl_->err; }
bool Enhancer::process(const FrameIn& in, uint8_t* dst, int stride) { return impl_->process(in, dst, stride); }
void Enhancer::reset() {
  Impl& s = *impl_;
  s.first = true; s.frameNo = 0; s.haveState = false; s.tileReset = true; s.lutBuilt = false;
  s.cur = s.ema = s.cfg.manual;
}
bool Enhancer::addLut(const Lut3D& l) {
  Impl& s = *impl_;
  if (l.size < 2 || l.data.size() != (size_t)l.size * l.size * l.size * 3 || s.bank.size() >= (size_t)kMaxLuts)
    return false;
  s.bank.push_back(l);
  s.lutBuilt = false;
  return true;
}
void Enhancer::setPredictor(std::shared_ptr<Predictor> p) {
  std::lock_guard<std::mutex> lk(impl_->predMx);
  impl_->pred = std::move(p);
  impl_->forceAnalysis = true;
}
void Enhancer::setManual(const Params& p) {
  { std::lock_guard<std::mutex> lk(impl_->cfgMx); impl_->liveCfg.manual = p; }
  impl_->cfg.manual = p;
  impl_->forceAnalysis = true;
}
Params Enhancer::currentParams() const { return impl_->cur; }
Params Enhancer::targetParams() const { return impl_->ema; }
Stats Enhancer::stats() const { return impl_->st; }
const Config& Enhancer::config() const { return impl_->cfg; }

}  // namespace hie
