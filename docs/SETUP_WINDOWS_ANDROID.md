# Setup & real-device testing guide (Windows and Android)

You do **not** need to know C/C++. You only copy/paste commands. Every step says what to expect.

> **Honest status:** this engine was built and tested (18 tests, 75 checks) on Linux x86-64.
> It has **not yet been built on Windows or on an Android/ARM phone** by its author. Your run is the
> first real test. If something fails, copy the error text and send it to me - that is expected and fixable.

Three things you can test, from easiest to most involved:

| Test | What it proves |
|---|---|
| 1. `hie_tests` | the maths is correct on your hardware (colour accuracy, determinism) |
| 2. `hie_cli bench` / `enhance` | speed on your CPU, and a before/after of your own photos |
| 3. `webcam_preview.py` (Windows) | live webcam through the engine |

---

## Part A - Windows 10/11

### A1. Install the compiler (one time, ~10 min)
1. Go to <https://visualstudio.microsoft.com/downloads/> and download **Build Tools for Visual Studio 2022**
   (or "Visual Studio 2022 Community" - free).
2. In the installer tick **"Desktop development with C++"** and click Install. This includes CMake.
3. Optional (only for test 3): install Python from <https://www.python.org/downloads/> - tick **"Add python.exe to PATH"**.

### A2. Get the code
Extract `hie_engine.tar.gz` (Windows 10/11 can do it: open **PowerShell** in the folder holding the file and run
`tar -xzf hie_engine.tar.gz`). Then:
```
cd hie
```

### A3. Build
Open **"x64 Native Tools Command Prompt for VS 2022"** from the Start menu (important: not a normal prompt), then:
```
cd C:\path\to\hie
cmake -S . -B build
cmake --build build --config Release
```
Expect a lot of text ending with `hie_tests.vcxproj -> ...Release\hie_tests.exe` and no `error`.
Warnings are fine. Output goes to `build\Release\`.

*Faster build for modern PCs (2013+ CPUs with AVX2):* add `-DHIE_WIN_AVX2=ON` to the first command.
Default is OFF so it also runs on old CPUs.

### A4. Test 1 - correctness
```
build\Release\hie_tests.exe
```
Last line should be `18 tests, 75 checks, 0 failed`.

### A5. Test 2 - speed and your own photo
```
build\Release\hie_cli.exe bench --sweep
build\Release\hie_cli.exe enhance C:\path\photo.jpg C:\path\photo_out.png
```
Open both images side by side. `bench` prints ms per frame at several resolutions; on Linux a 2-core
server did 1080p in ~7.6 ms. Please send me your numbers plus your CPU model (Task Manager > Performance).
Note: on MSVC the 3D-LUT path runs scalar (no runtime CPU dispatch), so it may be slower than Linux; the default
colour path is unaffected.

### A6. Test 3 - live webcam
```
pip install opencv-python numpy
python tools\webcam_preview.py --selftest
python tools\webcam_preview.py
```
`--selftest` needs no camera and should print `SELFTEST OK`. The live window shows engine time and fps;
**SPACE** toggles original/enhanced, **Q** quits. If the wrong camera opens use `--camera 1`.
To test one picture instead: `python tools\webcam_preview.py --image photo.jpg`.

### Windows troubleshooting
- `'cmake' is not recognized` -> you used a normal prompt; use the *x64 Native Tools* prompt.
- `Could not find hie_c.dll` -> run the build first, or `--lib build\Release`.
- Window shows black / camera error -> close Teams/Zoom/browser tabs using the camera.

---

## Part B - Android (phone/tablet, no PC needed)

The easiest real-device route is **Termux**, a Linux terminal app. It also gives the first real test of the
ARM (NEON) speed path.

### B1. Install Termux
Install from **F-Droid** (<https://f-droid.org/packages/com.termux/>), **not** the Play Store (that version is outdated and its
package system is broken).

### B2. Install the tools (one time)
In Termux:
```
pkg update
pkg install clang cmake make tar
termux-setup-storage
```
Accept the storage permission popup.

### B3. Get the code onto the phone
Copy `hie_engine.tar.gz` to the phone's **Download** folder (USB, cloud drive, etc.), then:
```
cd ~
tar -xzf ~/storage/downloads/hie_engine.tar.gz
cd hie
```

### B4. Build (takes a few minutes on a phone)
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```
(The Linux webcam tool is skipped automatically on Android.)

### B5. Tests
```
./build/hie_tests
./build/hie_cli bench --sweep
```
Expect `18 tests, 75 checks, 0 failed`. Send me the bench numbers and the phone model.

### B6. Enhance your own phone photos
```
cp ~/storage/dcim/Camera/IMG_0001.jpg in.jpg
./build/hie_cli enhance in.jpg out.png
cp out.png ~/storage/pictures/
```
Open `out.png` in the Gallery next to the original. (File names under `dcim/Camera` vary by phone: `ls ~/storage/dcim/Camera`.)

### Android troubleshooting
- Build runs out of memory -> use `cmake --build build -j1`.
- `Permission denied` on storage -> run `termux-setup-storage` again, restart Termux.
- Termux has no easy OpenCV, so the live-webcam script is Windows/Linux/macOS only for now.

### What is NOT covered yet
A real Android **camera app** (Android Studio + NDK + JNI/Camera2) is a separate, larger step that I have not written.
Termux gives the speed/correctness result first; if it passes on your phone I can build a sample app next.

---

## What to send back
1. Pass/fail of each step and any error text (copy/paste or screenshot).
2. `bench --sweep` output + CPU/phone model.
3. A before/after photo pair where the result looked wrong, if any.
