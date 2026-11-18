# Accuracy notes

* **Why not only a 17³ LUT?** The spec's fused 17³ LUT samples a non-linear transform. Measured against the exact float
  transform (random colours + neutral axis, 4 parameter sets): 17³ tetrahedral mean ΔE00 0.089, p99.9 1.78, worst 10 levels
  (clipping kinks); 33³: mean 0.026, p99.9 1.0. The default colour stage therefore evaluates the analytic transform
  *exactly* via 3 per-channel tables + 1 luma-indexed table (4 lookups/pixel): mean error 0.008 level, max 1 level.
  The 3D LUT (tetrahedral by default, trilinear optional, `lut_size` 17/33) is used for external `.cube` looks and learned
  LUT weights, or when `Config::prefer_3d_lut` is set.
* Shaper (`shaper_gamma`) sweep on the 17³ grid: 1.0 was best (0.089) vs 0.8 (0.126) and 0.65 (0.128); default is 1.0.
* White balance and exposure are applied in linear light, WB gains are luminance-normalised, CLAHE output is confined to the
  legal luma range (16–235 for limited range) so it never pushes colours out of gamut.
* Tetrahedral SIMD (SSE4.1) and scalar kernels are bit-identical (tested).
* AVX2 gather kernels were measured *slower* than scalar on the test CPU and were removed.
