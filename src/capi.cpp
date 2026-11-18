#include "hie/hie_c.h"

#include <new>

#include "hie/hie.h"

struct hie_handle {
  hie::Enhancer enh;
  explicit hie_handle(const hie::Config& c) : enh(c) {}
};

extern "C" {

HIE_API void hie_config_default(hie_config* c, int width, int height) {
  if (!c) return;
  *c = hie_config{};
  c->width = width; c->height = height;
  c->matrix = 0; c->range = 0;
  c->analysis_interval = 12; c->async_analysis = 1; c->lut_size = 17;
  c->enable_denoise = c->enable_clahe = c->enable_lut = 1;
  c->auto_wb = 1;
}

HIE_API hie_handle* hie_create(const hie_config* c) {
  if (!c) return nullptr;
  hie::Config cfg;
  cfg.width = c->width; cfg.height = c->height;
  cfg.in_format = (hie::InFormat)c->in_format;
  cfg.out_format = (hie::OutFormat)c->out_format;
  cfg.matrix = c->matrix ? hie::Matrix::BT709 : hie::Matrix::BT601;
  cfg.range = c->range ? hie::Range::Full : hie::Range::Limited;
  cfg.threads = c->threads;
  if (c->analysis_interval > 0) cfg.analysis_interval = c->analysis_interval;
  cfg.async_analysis = c->async_analysis != 0;
  if (c->lut_size > 0) cfg.lut_size = c->lut_size;
  cfg.enable_denoise = c->enable_denoise != 0;
  cfg.enable_clahe = c->enable_clahe != 0;
  cfg.enable_lut = c->enable_lut != 0;
  cfg.auto_wb = c->auto_wb != 0;
  hie_handle* h = new (std::nothrow) hie_handle(cfg);
  if (h && !h->enh.valid()) { delete h; return nullptr; }
  return h;
}

HIE_API void hie_destroy(hie_handle* h) { delete h; }

HIE_API int hie_process(hie_handle* h, const uint8_t* const planes[3], const int strides[3],
                        uint8_t* dst, int dst_stride) {
  if (!h || !planes || !strides) return 0;
  hie::FrameIn in;
  for (int i = 0; i < 3; ++i) { in.data[i] = planes[i]; in.stride[i] = strides[i]; }
  return h->enh.process(in, dst, dst_stride) ? 1 : 0;
}

HIE_API int hie_load_cube_lut(hie_handle* h, const char* path) {
  if (!h || !path) return 0;
  hie::Lut3D l;
  if (!hie::Lut3D::loadCube(path, l)) return 0;
  return h->enh.addLut(l) ? 1 : 0;
}

HIE_API void hie_reset(hie_handle* h) { if (h) h->enh.reset(); }

HIE_API void hie_get_params(const hie_handle* h, float out[9]) {
  if (!h || !out) return;
  hie::Params p = h->enh.currentParams();
  out[0] = p.wb_r; out[1] = p.wb_b; out[2] = p.exposure_ev; out[3] = p.contrast;
  out[4] = p.saturation; out[5] = p.shadow_lift; out[6] = p.clahe_strength;
  out[7] = p.clahe_clip; out[8] = p.denoise_t;
}

HIE_API const char* hie_version(void) { return HIE_VERSION_STRING; }

}  // extern "C"
