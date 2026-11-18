// hie_webcam - Linux virtual webcam: V4L2 capture (YUYV) -> HIE -> v4l2loopback output (YUYV).
//   sudo modprobe v4l2loopback video_nr=10 card_label="HIE Cam" exclusive_caps=1
//   hie_webcam --in /dev/video0 --out /dev/video10 --width 1280 --height 720
// Zoom / Teams / Meet / OBS then select "HIE Cam".
// NOTE: written against the V4L2 API but NOT exercised on real hardware in the build environment.
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hie/hie.h"

static volatile std::sig_atomic_t g_stop = 0;
static void onSig(int) { g_stop = 1; }

static int xioctl(int fd, unsigned long req, void* arg) {
  int r;
  do r = ioctl(fd, req, arg); while (r == -1 && errno == EINTR);
  return r;
}

// RGB24 -> YUYV (BT.601 limited), chroma averaged over horizontal pairs.
static void rgbToYuyv(const uint8_t* rgb, uint8_t* out, int w, int h) {
  for (int y = 0; y < h; ++y) {
    const uint8_t* s = rgb + (size_t)y * w * 3;
    uint8_t* d = out + (size_t)y * w * 2;
    for (int x = 0; x < w; x += 2) {
      int Y[2], U = 0, V = 0;
      for (int k = 0; k < 2; ++k) {
        int r = s[3 * (x + k)], g = s[3 * (x + k) + 1], b = s[3 * (x + k) + 2];
        Y[k] = 16 + ((16829 * r + 33039 * g + 6416 * b + 32768) >> 16);
        U += 128 + ((-9714 * r - 19071 * g + 28784 * b + 32768) >> 16);
        V += 128 + ((28784 * r - 24103 * g - 4681 * b + 32768) >> 16);
      }
      d[2 * x] = (uint8_t)Y[0]; d[2 * x + 1] = (uint8_t)(U / 2);
      d[2 * x + 2] = (uint8_t)Y[1]; d[2 * x + 3] = (uint8_t)(V / 2);
    }
  }
}

int main(int argc, char** argv) {
  std::string in = "/dev/video0", out = "/dev/video10";
  int w = 1280, h = 720;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k == "--in") in = argv[i + 1];
    else if (k == "--out") out = argv[i + 1];
    else if (k == "--width") w = std::atoi(argv[i + 1]);
    else if (k == "--height") h = std::atoi(argv[i + 1]);
  }
  std::signal(SIGINT, onSig); std::signal(SIGTERM, onSig);

  int cap = open(in.c_str(), O_RDWR);
  if (cap < 0) { std::perror(("open " + in).c_str()); return 1; }
  v4l2_format f{};
  f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  f.fmt.pix.width = w; f.fmt.pix.height = h;
  f.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; f.fmt.pix.field = V4L2_FIELD_NONE;
  if (xioctl(cap, VIDIOC_S_FMT, &f) < 0 || f.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
    std::fprintf(stderr, "camera does not offer YUYV at %dx%d (MJPEG-only modes are not supported)\n", w, h);
    return 1;
  }
  w = f.fmt.pix.width; h = f.fmt.pix.height;
  const int cstride = f.fmt.pix.bytesperline;

  v4l2_requestbuffers rb{};
  rb.count = 4; rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; rb.memory = V4L2_MEMORY_MMAP;
  if (xioctl(cap, VIDIOC_REQBUFS, &rb) < 0) { std::perror("REQBUFS"); return 1; }
  std::vector<void*> maps(rb.count); std::vector<size_t> lens(rb.count);
  for (unsigned i = 0; i < rb.count; ++i) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = i;
    if (xioctl(cap, VIDIOC_QUERYBUF, &b) < 0) { std::perror("QUERYBUF"); return 1; }
    lens[i] = b.length;
    maps[i] = mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, cap, b.m.offset);
    if (maps[i] == MAP_FAILED) { std::perror("mmap"); return 1; }
    if (xioctl(cap, VIDIOC_QBUF, &b) < 0) { std::perror("QBUF"); return 1; }
  }
  v4l2_buf_type t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(cap, VIDIOC_STREAMON, &t) < 0) { std::perror("STREAMON"); return 1; }

  int sink = open(out.c_str(), O_WRONLY);
  if (sink < 0) { std::perror(("open " + out).c_str()); return 1; }
  v4l2_format of{};
  of.type = V4L2_BUF_TYPE_VIDEO_OUTPUT;
  of.fmt.pix.width = w; of.fmt.pix.height = h;
  of.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; of.fmt.pix.field = V4L2_FIELD_NONE;
  of.fmt.pix.bytesperline = w * 2; of.fmt.pix.sizeimage = w * h * 2;
  if (xioctl(sink, VIDIOC_S_FMT, &of) < 0) { std::perror("S_FMT (loopback)"); return 1; }

  hie::Config cfg;
  cfg.width = w; cfg.height = h; cfg.in_format = hie::InFormat::YUYV;
  cfg.out_format = hie::OutFormat::RGB24;
  cfg.matrix = hie::Matrix::BT601;
  hie::Enhancer enh(cfg);
  if (!enh.valid()) { std::fprintf(stderr, "%s\n", enh.error().c_str()); return 1; }

  hie::FrameRing rgbRing(2, (size_t)w * h * 3);
  std::vector<uint8_t> yuyv((size_t)w * h * 2);
  std::fprintf(stderr, "hie_webcam: %s -> %s  %dx%d\n", in.c_str(), out.c_str(), w, h);
  while (!g_stop) {
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP;
    if (xioctl(cap, VIDIOC_DQBUF, &b) < 0) { if (errno == EAGAIN) continue; std::perror("DQBUF"); break; }
    hie::FrameIn fin;
    fin.data[0] = (const uint8_t*)maps[b.index]; fin.stride[0] = cstride;
    uint8_t* rgb = rgbRing.acquire();
    if (enh.process(fin, rgb, w * 3)) {
      rgbToYuyv(rgb, yuyv.data(), w, h);
      size_t off = 0;
      while (off < yuyv.size()) {
        ssize_t n = write(sink, yuyv.data() + off, yuyv.size() - off);
        if (n < 0) { if (errno == EINTR) continue; std::perror("write"); g_stop = 1; break; }
        off += (size_t)n;
      }
    }
    if (xioctl(cap, VIDIOC_QBUF, &b) < 0) { std::perror("QBUF"); break; }
  }
  xioctl(cap, VIDIOC_STREAMOFF, &t);
  for (unsigned i = 0; i < rb.count; ++i) munmap(maps[i], lens[i]);
  close(cap); close(sink);
  return 0;
}
