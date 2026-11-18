/* C API for JNI / Swift / Python ctypes / virtual-camera hosts. Mirrors hie.h. */
#ifndef HIE_C_H
#define HIE_C_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define HIE_API __declspec(dllexport)
#else
#  define HIE_API __attribute__((visibility("default")))
#endif

typedef struct hie_handle hie_handle;

/* in_format: 0=I420 1=NV12 2=YUYV 3=RGB24     out_format: 0=RGB24 1=BGR24 2=RGBA32 3=BGRA32
   matrix: 0=BT601 1=BT709                      range: 0=limited 1=full */
typedef struct hie_config {
  int width, height;
  int in_format, out_format, matrix, range;
  int threads;           /* 0 = auto */
  int analysis_interval; /* 0 = default (12) */
  int async_analysis;    /* 1 = background thread */
  int lut_size;          /* 0 = default (17) */
  int enable_denoise, enable_clahe, enable_lut;
  int auto_wb;
} hie_config;

HIE_API void hie_config_default(hie_config* c, int width, int height);
HIE_API hie_handle* hie_create(const hie_config* c);   /* NULL on failure */
HIE_API void hie_destroy(hie_handle* h);
/* planes/strides follow FrameIn: I420 {Y,U,V}, NV12 {Y,UV}, YUYV/RGB24 {packed}. Returns 1 on success. */
HIE_API int hie_process(hie_handle* h, const uint8_t* const planes[3], const int strides[3],
                        uint8_t* dst, int dst_stride);
HIE_API int hie_load_cube_lut(hie_handle* h, const char* path);
HIE_API void hie_reset(hie_handle* h);
/* out[0..] = wb_r, wb_b, exposure_ev, contrast, saturation, shadow_lift, clahe_strength, clahe_clip, denoise_t */
HIE_API void hie_get_params(const hie_handle* h, float out[9]);
HIE_API const char* hie_version(void);

#ifdef __cplusplus
}
#endif
#endif
