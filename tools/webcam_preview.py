#!/usr/bin/env python3
"""Live webcam (or single-image) preview through the HIE engine. Works on Windows, Linux, macOS.

  pip install opencv-python numpy
  python tools/webcam_preview.py                 # live webcam, press Q to quit, SPACE toggles enhancement
  python tools/webcam_preview.py --image in.jpg  # enhance one picture -> in_enhanced.png
  python tools/webcam_preview.py --selftest      # no camera needed: checks the library loads and runs
"""
import argparse, ctypes, os, sys, time
import numpy as np

class Cfg(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in (
        "width height in_format out_format matrix range threads analysis_interval async_analysis "
        "lut_size enable_denoise enable_clahe enable_lut auto_wb").split()]

def find_lib(explicit):
    names = ["hie_c.dll", "libhie_c.so", "libhie_c.dylib"]
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    dirs = [explicit] if explicit else []
    dirs += [os.getcwd(), root] + [os.path.join(root, d) for d in
            ("build", os.path.join("build", "Release"), os.path.join("build", "Debug"))]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p): return p
    sys.exit("Could not find hie_c.dll / libhie_c.so. Build the project first (see docs/SETUP_WINDOWS_ANDROID.md) "
             "or pass --lib <folder>.")

class Engine:
    def __init__(self, libdir, w, h):
        path = find_lib(libdir)
        if hasattr(os, "add_dll_directory"): os.add_dll_directory(os.path.dirname(path))
        L = ctypes.CDLL(path)
        L.hie_create.restype = ctypes.c_void_p
        L.hie_create.argtypes = [ctypes.POINTER(Cfg)]
        L.hie_destroy.argtypes = [ctypes.c_void_p]
        L.hie_version.restype = ctypes.c_char_p
        L.hie_process.restype = ctypes.c_int
        L.hie_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int]
        self.L, self.w, self.h = L, w, h
        c = Cfg()
        L.hie_config_default(ctypes.byref(c), w, h)
        c.in_format = 3      # RGB24 in
        c.out_format = 1     # BGR24 out
        self.h_ = L.hie_create(ctypes.byref(c))
        if not self.h_: sys.exit("hie_create failed")
        self.out = np.empty((h, w, 3), np.uint8)
        print("HIE version:", L.hie_version().decode(), "| library:", path)

    def process(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        planes = (ctypes.c_void_p * 3)(rgb.ctypes.data, None, None)
        strides = (ctypes.c_int * 3)(self.w * 3, 0, 0)
        ok = self.L.hie_process(self.h_, planes, strides, self.out.ctypes.data, self.w * 3)
        if not ok: raise RuntimeError("hie_process failed")
        return self.out

    def close(self): self.L.hie_destroy(self.h_)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--image")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--lib", help="folder containing hie_c.dll / libhie_c.so")
    a = ap.parse_args()
    import cv2

    if a.selftest:
        w, h = 640, 480
        e = Engine(a.lib, w, h)
        rng = np.random.default_rng(1)
        img = np.clip(rng.normal([150, 120, 80], 25, (h, w, 3)), 0, 255).astype(np.uint8)  # warm, noisy
        t = time.perf_counter()
        for _ in range(30): out = e.process(img)
        ms = (time.perf_counter() - t) * 1000 / 30
        print(f"SELFTEST OK  {w}x{h}: {ms:.2f} ms/frame, output mean BGR = {out.reshape(-1,3).mean(0).round(1)}")
        e.close(); return

    if a.image:
        bgr = cv2.imread(a.image)
        if bgr is None: sys.exit("cannot read " + a.image)
        h, w = bgr.shape[:2]
        e = Engine(a.lib, w, h)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for _ in range(12): out = e.process(rgb)       # let the auto analysis settle
        dst = os.path.splitext(a.image)[0] + "_enhanced.png"
        cv2.imwrite(dst, out); print("wrote", dst); e.close(); return

    w, h = map(int, a.size.lower().split("x"))
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened(): sys.exit("cannot open camera %d (try --camera 1)" % a.camera)
    ok, f = cap.read()
    if not ok: sys.exit("camera gave no frames")
    h, w = f.shape[:2]
    e = Engine(a.lib, w, h)
    on, fps, last = True, 0.0, time.perf_counter()
    while True:
        ok, bgr = cap.read()
        if not ok: break
        t0 = time.perf_counter()
        out = e.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)) if on else bgr
        ms = (time.perf_counter() - t0) * 1000
        now = time.perf_counter(); fps = 0.9 * fps + 0.1 / max(now - last, 1e-6); last = now
        view = out.copy()
        cv2.putText(view, f"{'ENHANCED' if on else 'ORIGINAL'}  engine {ms:.1f} ms  {fps:.0f} fps  (SPACE toggle, Q quit)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("HIE preview", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27): break
        if k == 32: on = not on
    e.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
