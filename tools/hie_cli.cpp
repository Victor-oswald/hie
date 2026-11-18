// hie_cli - still-image enhancement, benchmark and info.
//   hie_cli enhance in.(png|jpg|ppm) out.(png|jpg|ppm) [options]
//   hie_cli bench   [--width 1920 --height 1080 --frames 300 --in i420|nv12|yuyv|rgb24 ...]
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "hie/hie.h"
#include "synth.hpp"

using namespace hie;

namespace {

struct Args {
  std::vector<std::string> pos;
  std::map<std::string, std::string> opt;
  bool has(const char* k) const { return opt.count(k) > 0; }
  std::string get(const char* k, const std::string& d = "") const {
    auto it = opt.find(k);
    return it == opt.end() ? d : it->second;
  }
  float f(const char* k, float d) const { return has(k) ? (float)std::atof(opt.at(k).c_str()) : d; }
  int i(const char* k, int d) const { return has(k) ? std::atoi(opt.at(k).c_str()) : d; }
};

Args parse(int argc, char** argv, int from) {
  Args a;
  static const char* flags[] = {"no-denoise", "no-clahe", "no-lut", "tri", "no-async", "no-wb",
                                "bgr", "rgba", "sweep", "quiet"};
  for (int i = from; i < argc; ++i) {
    std::string s = argv[i];
    if (s.size() > 2 && s[0] == '-' && s[1] == '-') {
      std::string k = s.substr(2);
      bool isFlag = false;
      for (auto f : flags) if (k == f) isFlag = true;
      if (isFlag || i + 1 >= argc) a.opt[k] = "1";
      else a.opt[k] = argv[++i];
    } else a.pos.push_back(s);
  }
  return a;
}

bool endsWith(const std::string& s, const char* e) {
  size_t n = std::strlen(e);
  return s.size() >= n && s.compare(s.size() - n, n, e) == 0;
}

bool writePPM(const std::string& p, const uint8_t* d, int w, int h) {
  FILE* f = std::fopen(p.c_str(), "wb");
  if (!f) return false;
  std::fprintf(f, "P6\n%d %d\n255\n", w, h);
  std::fwrite(d, 1, (size_t)w * h * 3, f);
  std::fclose(f);
  return true;
}

void applyCommon(const Args& a, Config& c) {
  c.threads = a.i("threads", 0);
  if (a.has("no-denoise")) c.enable_denoise = false;
  if (a.has("no-clahe")) c.enable_clahe = false;
  if (a.has("no-lut")) c.enable_lut = false;
  if (a.has("tri")) c.tetrahedral = false;
  if (a.has("no-wb")) c.auto_wb = false;
  c.lut_size = a.i("lut-size", c.lut_size);
  c.shaper_gamma = a.f("shaper", c.shaper_gamma);
  c.wb_strength = a.f("wb-strength", c.wb_strength);
  c.exposure_strength = a.f("exposure-strength", c.exposure_strength);
  c.clahe_strength_max = a.f("clahe-strength", c.clahe_strength_max);
  c.clahe_tiles_x = c.clahe_tiles_y = a.i("tiles", c.clahe_tiles_x);
  c.clahe_hist_stride = a.i("hist-stride", c.clahe_hist_stride);
  if (a.get("matrix") == "709") c.matrix = Matrix::BT709;
  if (a.get("range") == "full") c.range = Range::Full;
  if (a.get("chroma") == "nearest") c.chroma_upsample = ChromaUpsample::Nearest;
  if (a.has("ev")) { c.auto_exposure = false; c.manual.exposure_ev = a.f("ev", 0); }
  if (a.has("contrast")) { c.auto_contrast = false; c.manual.contrast = a.f("contrast", 1); }
  if (a.has("sat")) { c.auto_saturation = false; c.manual.saturation = a.f("sat", 1); }
  if (a.has("lift")) { c.auto_shadows = false; c.manual.shadow_lift = a.f("lift", 0); }
}

void printParams(const Params& p) {
  std::printf("  wb gains r/b      : %.3f / %.3f\n", p.wb_r, p.wb_b);
  std::printf("  exposure          : %+.2f EV\n", p.exposure_ev);
  std::printf("  contrast / sat    : %.3f / %.3f\n", p.contrast, p.saturation);
  std::printf("  shadow lift       : %.3f\n", p.shadow_lift);
  std::printf("  CLAHE strength/clip: %.3f / %.2f\n", p.clahe_strength, p.clahe_clip);
  std::printf("  denoise threshold : %.1f\n", p.denoise_t);
}

int cmdEnhance(const Args& a) {
  if (a.pos.size() < 2) { std::fprintf(stderr, "usage: hie_cli enhance in out [options]\n"); return 2; }
  int w, h, n;
  uint8_t* img = stbi_load(a.pos[0].c_str(), &w, &h, &n, 3);
  if (!img) { std::fprintf(stderr, "cannot read %s: %s\n", a.pos[0].c_str(), stbi_failure_reason()); return 1; }

  Config c;
  c.width = w; c.height = h;
  c.in_format = InFormat::RGB24;
  c.out_format = OutFormat::RGB24;
  c.async_analysis = false;     // stills: deterministic, immediately applied
  c.frame_smoothing = 1.f;
  applyCommon(a, c);

  Enhancer e(c);
  if (!e.valid()) { std::fprintf(stderr, "error: %s\n", e.error().c_str()); return 1; }
  int slot = 0;
  for (const auto& kv : a.opt) {
    if (kv.first != "cube") continue;
    std::string spec = kv.second;
    float wgt = 1.f;
    size_t colon = spec.rfind(':');
    if (colon != std::string::npos && colon > 1) { wgt = (float)std::atof(spec.c_str() + colon + 1); spec.resize(colon); }
    Lut3D l; std::string err;
    if (!Lut3D::loadCube(spec, l, &err)) { std::fprintf(stderr, "cube: %s\n", err.c_str()); return 1; }
    if (!e.addLut(l) || slot >= kMaxLuts) { std::fprintf(stderr, "cube: too many LUTs\n"); return 1; }
    Params m = c.manual; m.lut_weights[slot++] = wgt;
    e.setManual(m);
  }

  std::vector<uint8_t> out((size_t)w * h * 3);
  FrameIn in = FrameIn::contiguous(InFormat::RGB24, w, h, img);
  e.process(in, out.data(), w * 3);          // frame 0 analyses inline
  auto t0 = std::chrono::steady_clock::now();
  e.process(in, out.data(), w * 3);          // frame 1 = steady-state cost
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  const std::string& o = a.pos[1];
  bool ok;
  if (endsWith(o, ".png")) ok = stbi_write_png(o.c_str(), w, h, 3, out.data(), w * 3);
  else if (endsWith(o, ".jpg") || endsWith(o, ".jpeg")) ok = stbi_write_jpg(o.c_str(), w, h, 3, out.data(), 95);
  else ok = writePPM(o, out.data(), w, h);
  if (!ok) { std::fprintf(stderr, "cannot write %s\n", o.c_str()); return 1; }
  if (!a.has("quiet")) {
    std::printf("%s -> %s  (%dx%d)  steady-state frame: %.2f ms\n", a.pos[0].c_str(), o.c_str(), w, h, ms);
    printParams(e.currentParams());
  }
  stbi_image_free(img);
  return 0;
}

InFormat parseIn(const std::string& s) {
  if (s == "nv12") return InFormat::NV12;
  if (s == "yuyv") return InFormat::YUYV;
  if (s == "rgb24") return InFormat::RGB24;
  return InFormat::I420;
}

int benchOne(const Args& a, int w, int h) {
  Config c;
  c.width = w; c.height = h;
  c.in_format = parseIn(a.get("in", "i420"));
  c.out_format = a.has("rgba") ? OutFormat::RGBA32 : (a.has("bgr") ? OutFormat::BGR24 : OutFormat::RGB24);
  c.async_analysis = !a.has("no-async");
  applyCommon(a, c);
  if (c.matrix == Matrix::BT601 && w >= 1280) c.matrix = Matrix::BT709;   // HD default
  if (a.get("matrix") == "601") c.matrix = Matrix::BT601;
  const int frames = a.i("frames", 300), warm = 30, nvar = 6;

  // dim, colour-cast, noisy webcam-like scene so every stage (denoise, CLAHE, WB/exposure LUT) is live
  const float cast[3] = {1.25f, 1.0f, 0.75f};
  std::vector<std::vector<uint8_t>> src;
  for (int i = 0; i < nvar; ++i)
    src.push_back(synth::rgbToFormat(synth::sceneRGB(w, h, 100 + i, 0.40f, cast, 4.0f), w, h, c.in_format, c.matrix, c.range));
  size_t outBytes = (size_t)w * h * 4;
  FrameRing ring(3, outBytes);
  const int ostride = w * ((c.out_format == OutFormat::RGB24 || c.out_format == OutFormat::BGR24) ? 3 : 4);

  Enhancer e(c);
  if (!e.valid()) { std::fprintf(stderr, "error: %s\n", e.error().c_str()); return 1; }
  std::vector<double> t;
  double sp = 0, sd = 0, sc = 0, so = 0;
  for (int i = 0; i < frames + warm; ++i) {
    FrameIn in = FrameIn::contiguous(c.in_format, w, h, src[i % nvar].data());
    auto t0 = std::chrono::steady_clock::now();
    e.process(in, ring.acquire(), ostride);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (i >= warm) {
      t.push_back(ms);
      Stats s = e.stats();
      sp += s.ms_prep; sd += s.ms_denoise; sc += s.ms_clahe; so += s.ms_output;
    }
  }
  std::sort(t.begin(), t.end());
  double mean = 0; for (double v : t) mean += v; mean /= t.size();
  auto pct = [&](double q) { return t[std::min(t.size() - 1, (size_t)(q * t.size()))]; };
  const char* names[] = {"I420", "NV12", "YUYV", "RGB24"};
  std::printf("%4dx%-4d %-5s thr=%d  mean %6.2f ms  p50 %6.2f  p95 %6.2f  p99 %6.2f  max %6.2f  -> %6.1f fps"
              "   [prep %.2f  denoise %.2f  clahe %.2f  output %.2f]\n",
              w, h, names[(int)c.in_format], a.i("threads", 0), mean, pct(.5), pct(.95), pct(.99), t.back(),
              1000.0 / mean, sp / t.size(), sd / t.size(), sc / t.size(), so / t.size());
  if (!a.has("quiet")) printParams(e.currentParams());
  return 0;
}

int cmdBench(const Args& a) {
  if (a.has("sweep")) {
    const int dims[][2] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    for (auto& d : dims) if (benchOne(a, d[0], d[1])) return 1;
    return 0;
  }
  return benchOne(a, a.i("width", 1920), a.i("height", 1080));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("hie_cli %s\n  enhance in out [--no-denoise --no-clahe --no-lut --tri --lut-size N --ev X --contrast X\n"
                "          --sat X --lift X --wb-strength X --no-wb --cube file[:weight] --threads N]\n"
                "  bench   [--width W --height H --frames N --in i420|nv12|yuyv|rgb24 --rgba --threads N --no-async --sweep]\n",
                HIE_VERSION_STRING);
    return 0;
  }
  std::string cmd = argv[1];
  Args a = parse(argc, argv, 2);
  if (cmd == "enhance") return cmdEnhance(a);
  if (cmd == "bench") return cmdBench(a);
  std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
  return 2;
}
