# Cycling Runtime Matrix

Measured on 2026-08-07. This document records local command output only. It
does not claim that a runtime combination is supported merely because its
files are present.

## Repository Baseline

| Component | Repository baseline | Source |
| --- | --- | --- |
| OpenCV | >= 4.6 | `CMakeLists.txt:3` |
| GStreamer | >= 1.14.5 | `CMakeLists.txt:4` |
| CUDA | 11.1 | `third_party/trt_yolov8/README.md:44` |
| TensorRT | 8.5 | `third_party/trt_yolov8/README.md:44` |
| PaddleOCR C++ | Derived from PaddleOCR `release/2.6/deploy/cpp_infer` | `third_party/paddle_ocr/README.md:2` |

CUDA, TensorRT, and Paddle are optional in the root build. The repository
baseline above is a historical tested pair, not proof that the current host
or WSL installation matches it.

## Measured Environments

### Windows Host

| Item | Exact command | Observed result |
| --- | --- | --- |
| OS | `Get-CimInstance Win32_OperatingSystem \| Select Caption,Version,BuildNumber,OSArchitecture` | Windows 11 Pro, version `10.0.26200`, build `26200`, 64-bit |
| GPU and driver | `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` | NVIDIA GeForce RTX 4070; driver `591.86` |
| Driver CUDA capability | `nvidia-smi` | `CUDA Version: 13.1`; this is the driver-reported capability, not the selected toolkit |
| Active CUDA toolkit | `Get-Command nvcc`; `nvcc --version` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe`; CUDA compilation tools `13.1`, `V13.1.80` |
| CUDA 13.1 runtime inventory | `$cudart='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cudart64_13.dll'; Test-Path $cudart; Get-Item $cudart` | `True`; `cudart64_13.dll` exists at the stated path, length `540784` bytes |
| Additional CUDA toolkit | `& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe' --version` | CUDA compilation tools `12.1`, `V12.1.66` |
| CUDA 12.1 runtime inventory | `$cudart='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\cudart64_12.dll'; Test-Path $cudart; Get-Item $cudart` | `True`; `cudart64_12.dll` exists at the stated path, length `527872` bytes |
| TensorRT files and version | `$header='D:\TensorRT-10.7\include\NvInferVersion.h'; $dll='D:\TensorRT-10.7\lib\nvinfer_10.dll'; $lib='D:\TensorRT-10.7\lib\nvinfer_10.lib'; $trtexec='D:\TensorRT-10.7\bin\trtexec.exe'; Test-Path $header; Test-Path $dll; Test-Path $lib; Test-Path $trtexec; Get-Item $dll,$lib,$trtexec; Select-String -Path $header -Pattern '^#define NV_TENSORRT_MAJOR ','^#define NV_TENSORRT_MINOR ','^#define NV_TENSORRT_PATCH ','^#define NV_TENSORRT_BUILD '` | All four `Test-Path` calls returned `True`; the header reports TensorRT `10.7.0.23`; `nvinfer_10.dll`, `nvinfer_10.lib`, and `trtexec.exe` exist under `D:\TensorRT-10.7` |
| `trtexec` availability | `Get-Command trtexec -ErrorAction SilentlyContinue`; `& 'D:\TensorRT-10.7\bin\trtexec.exe' --version`; `$LASTEXITCODE` | PATH lookup returned no command. The absolute-path binary started and printed `TensorRT v100700 [b23]`; the 2026-08-07 recheck returned `$LASTEXITCODE=1` because no model was supplied. No model or engine was loaded and inference was `NOT RUN`. |
| OpenCV | `Get-Command opencv_version`; `opencv_version` | `D:\opencv\opencv\build\x64\vc16\bin\opencv_version.exe`; version `4.10.0` |
| GStreamer core/app/RTSP | `Get-Command gst-launch-1.0,gst-inspect-1.0,pkg-config`; `Test-Path C:\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe` | `NOT FOUND`; the referenced GStreamer path does not contain the executable |
| Paddle Inference | `Get-Command paddle_inference.dll -ErrorAction SilentlyContinue`; `Test-Path -LiteralPath 'C:\paddle_inference'`; `Test-Path -LiteralPath 'D:\paddle_inference'`; `Test-Path -LiteralPath 'C:\Program Files\Paddle'`; `Test-Path -LiteralPath 'D:\Paddle'` | Command lookup returned no result; all four path checks returned `False`: `NOT FOUND` |
| CMake | `Get-Command cmake`; `& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --version` | Current PowerShell PATH: `NOT FOUND`; Visual Studio bundled CMake: `3.31.6-msvc6` |
| C++ compiler | `$vswhere='C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'; Test-Path $vswhere; & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`; `& 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe' /Bv` | `vswhere.exe` exists and returned Visual Studio 2022 Build Tools; MSVC `19.44.35221`, tool directory `14.44.35207` (`cl` then exits with the expected missing-source diagnostic) |

### WSL/Linux Build Environment

Distribution: Ubuntu `22.04.5 LTS` under WSL2, kernel
`6.6.87.2-microsoft-standard-WSL2`.

| Item | Exact command | Observed result |
| --- | --- | --- |
| OS/kernel | `uname -a`; `cat /etc/os-release` | Ubuntu `22.04.5 LTS`; x86_64 WSL2 kernel `6.6.87.2` |
| GPU and driver | `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`; `nvidia-smi` | NVIDIA GeForce RTX 4070; Windows driver `591.86`; driver-reported CUDA capability `13.1` |
| CUDA toolkit | `command -v nvcc`; `nvcc --version` | `NOT INSTALLED` / `NOT FOUND` |
| TensorRT / `trtexec` | `wsl.exe -- which trtexec`; `wsl.exe -- find /usr/local /opt -maxdepth 4 -type f -name trtexec`; `wsl.exe -- find /usr/local /opt -maxdepth 4 -type f -name libnvinfer.so*` | `which` exited `1` with no path; both `find` commands returned no files: `NOT INSTALLED` / `NOT FOUND` |
| OpenCV | `pkg-config --modversion opencv4`; `opencv_version` | `4.5.4`; executable `/usr/bin/opencv_version`; CMake package under `/usr/lib/x86_64-linux-gnu/cmake/opencv4` |
| GStreamer core | `pkg-config --modversion gstreamer-1.0` | `1.20.3` |
| GStreamer app | `pkg-config --modversion gstreamer-app-1.0` | `1.20.1` |
| GStreamer RTSP server | `pkg-config --modversion gstreamer-rtsp-server-1.0` | `1.20.1` |
| Paddle Inference | `wsl.exe -- test -d /usr/local/paddle_inference`; `wsl.exe -- find /usr/local /opt -maxdepth 4 -type f -name libpaddle_inference.so*` | Directory check exited `1`; `find` returned no files: `NOT INSTALLED` / `NOT FOUND` |
| CMake | `env -u CMAKE_ROOT /usr/bin/cmake --version` | `3.22.1` at `/usr/bin/cmake` |
| C/C++ compiler | `gcc --version`; `g++ --version` | GCC and G++ `11.4.0` at `/usr/bin/gcc` and `/usr/bin/g++` |

The explicit `/usr/bin/cmake` and `env -u CMAKE_ROOT` form was used for all
WSL configurations. A plain WSL interop invocation printed a spurious
`Could not find CMAKE_ROOT` error after its version output; the explicit form
configured both isolated builds successfully.

## Isolated Build Profiles

The root CMake project now accepts only these cache values:

- `VP_RUNTIME_PROFILE=baseline` (default)
- `VP_RUNTIME_PROFILE=yolo11-pilot`

Each configure writes `${build}/vp_runtime_profile.txt` with the profile,
CUDA/TensorRT/Paddle/YOLO11/async-OCR/test switches, and detected OpenCV and
GStreamer versions only after configure and generation succeed. A failed
configure removes any stale manifest and does not publish a replacement. The
profile does not install a dependency or mutate an existing runtime.

`yolo11-pilot` is an audit/profile label only. It does not automatically enable
YOLO11, CUDA, or TensorRT; the actual switches remain controlled by explicit
`-DVP_WITH_*=ON/OFF` arguments.

### Default and validation checks

The commands below are self-contained and must be run from the repository root.

```powershell
$repoWindows = (Resolve-Path '.').Path
$repoWsl = (wsl.exe -e wslpath -a $repoWindows).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($repoWsl)) {
    throw 'Failed to resolve the repository path in WSL.'
}

$baselineBuild = "$repoWsl/build-task6-baseline"
$invalidBuild = '/tmp/videopipe-task6-invalid-20260807'

wsl.exe -- env -u CMAKE_ROOT /usr/bin/cmake -S $repoWsl -B $baselineBuild -DVP_WITH_CUDA=OFF -DVP_WITH_TRT=OFF -DVP_WITH_PADDLE=OFF -DVP_WITH_YOLO11=OFF -DVP_WITH_ASYNC_OCR=OFF -DVP_BUILD_TESTS=ON
if ($LASTEXITCODE -ne 0) { throw 'Baseline configure failed.' }

wsl.exe -- env -u CMAKE_ROOT /usr/bin/cmake -S $repoWsl -B $invalidBuild -DVP_RUNTIME_PROFILE=invalid
if ($LASTEXITCODE -eq 0) { throw 'Invalid profile unexpectedly configured.' }
wsl.exe -- test ! -e "$invalidBuild/vp_runtime_profile.txt"
if ($LASTEXITCODE -ne 0) { throw 'Failed configure published a manifest.' }
```

Results:

- Default profile configured as `baseline` and generated the manifest.
- The invalid value failed at configure time with
  `Invalid VP_RUNTIME_PROFILE='invalid'. Allowed values: baseline, yolo11-pilot`.
- `build-task6-baseline/vp_runtime_profile.txt` records CUDA, TensorRT,
  Paddle, YOLO11, and async OCR as `OFF`; tests are `ON`.
- `vp_cmake_smoke` and `vp_bytetrack_test` built and passed independently.
- `nodes/vp_rtsp_src_node.o` and `nodes/vp_app_src_node.o` compiled against
  the detected GStreamer development packages.

### YOLO11 ONNX pilot

```powershell
$repoWindows = (Resolve-Path '.').Path
$repoWsl = (wsl.exe -e wslpath -a $repoWindows).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($repoWsl)) {
    throw 'Failed to resolve the repository path in WSL.'
}

$pilotBuild = "$repoWsl/build-task6-yolo11-pilot"
wsl.exe -- env -u CMAKE_ROOT /usr/bin/cmake -S $repoWsl -B $pilotBuild -DVP_RUNTIME_PROFILE=yolo11-pilot -DVP_WITH_CUDA=OFF -DVP_WITH_TRT=OFF -DVP_WITH_PADDLE=OFF -DVP_WITH_YOLO11=ON -DVP_WITH_ASYNC_OCR=OFF -DVP_BUILD_TESTS=ON
if ($LASTEXITCODE -ne 0) { throw 'YOLO11 pilot configure failed.' }

wsl.exe -- env -u CMAKE_ROOT /usr/bin/cmake --build $pilotBuild --target vp_yolo11_postprocess_test -- -j2
if ($LASTEXITCODE -ne 0) { throw 'YOLO11 postprocess build failed.' }

wsl.exe -- env -u CMAKE_ROOT /usr/bin/ctest --test-dir $pilotBuild -R vp_yolo11_postprocess_test --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'YOLO11 postprocess test failed.' }
```

Result: configure, build, and the synthetic YOLO11 postprocess test passed.
`VP_CYCLING_MODEL_DIR` is not set in Windows or WSL, and no ONNX/engine file
was found in the repository. Model loading, detector inference, ONNX/TRT box
comparison, latency, recall, and FPS are therefore `BLOCKED / NOT TESTED`.

## Measured Failure and Upgrade Guidance

The WSL OpenCV `4.5.4` installation is below the repository's `4.6` minimum.
The full shared-library build reproduced this error:

```text
nodes/infers/vp_face_swap_node.cpp:423:13: error: 'transposeND' is not a member of 'cv'; did you mean 'transpose'?
  423 |         cv::transposeND(output, {0,2,3,1}, output_channel_last);
```

This is a measured reason to provide an isolated OpenCV >= 4.6 build
environment. It is not a reason to change the source in this task.

WSL GStreamer core/app/RTSP versions all exceed the repository minimum. CMake
found their headers and libraries, and the RTSP/app-source objects compiled.
No GStreamer upgrade is recommended from current evidence. RTSP runtime,
hardware decode, recorder behavior, and reconnect endurance remain untested.

## TensorRT/CUDA Matched-Pair Gate

Windows has TensorRT `10.7.0.23`, CUDA toolkits `12.1` and `13.1`, and driver
`591.86`. That inventory is not a validated matched pair. WSL has neither a
CUDA toolkit nor TensorRT. No engine was built, and the old YOLOv8 builder was
not modified.

Future execution, only in an isolated pilot after selecting a TensorRT-supported
CUDA toolkit and providing the external model, should use commands equivalent
to:

```powershell
$modelDir = $env:VP_CYCLING_MODEL_DIR
if ([string]::IsNullOrWhiteSpace($modelDir)) {
    throw 'VP_CYCLING_MODEL_DIR is not set.'
}

$trtexec = 'D:\TensorRT-10.7\bin\trtexec.exe'
$onnx = Join-Path $modelDir 'yolo11s.onnx'
$engine = Join-Path $modelDir 'yolo11s-trt10.7.engine'
$output = Join-Path $modelDir 'yolo11s-trt-output.json'
if (-not (Test-Path -LiteralPath $trtexec)) { throw 'trtexec was not found.' }
if (-not (Test-Path -LiteralPath $onnx)) { throw 'YOLO11 ONNX model was not found.' }

nvcc --version
if ($LASTEXITCODE -ne 0) { throw 'CUDA toolkit check failed.' }
& $trtexec --onnx=$onnx --saveEngine=$engine --fp16 --skipInference
if ($LASTEXITCODE -ne 0) { throw 'TensorRT engine build failed.' }
& $trtexec --loadEngine=$engine --dumpOutput --exportOutput=$output
if ($LASTEXITCODE -ne 0) { throw 'TensorRT inference failed.' }
```

Promotion additionally requires comparing TensorRT boxes with the ONNX
reference on the same inputs. The commands above were not executed because
the model directory is absent and the installed CUDA/TensorRT pair has not
been qualified.

## PaddleOCR 3.x Pilot Gate

Paddle Inference is absent in both measured environments, and the fixed OCR
crop set is not configured. PaddleOCR 3.x is therefore `BLOCKED / NOT TESTED`.
The pilot must remain behind the existing `vp_ocr_job -> vp_ocr_consensus`
boundary and must be compared against the repository's PaddleOCR 2.6-derived
fallback on identical crops for recognition, p95 latency, exceptions, and
queue backlog. OCR failure must not delete an athlete observation or evidence.

`third_party/paddle_ocr/README.md` was intentionally not changed because no
new Paddle runtime was measured.

## Decision Table

| Candidate | Status on 2026-08-07 | Evidence | Promotion gate | Rollback |
| --- | --- | --- | --- | --- |
| Current baseline | `PARTIAL`: isolated profile and independent tests pass; full shared-library build blocked | baseline manifest; smoke/ByteTrack tests pass; WSL OpenCV 4.5.4 reproduces `transposeND` compile failure | Build full library with OpenCV >= 4.6, then run real clips and reconnect test without athlete-evidence loss | Keep all optional runtime switches `OFF` and use the existing YOLOv8/SORT/synchronous OCR path |
| YOLO11 ONNX pilot | `PARTIAL`: postprocess test passes; inference `BLOCKED / NOT TESTED` | yolo11-pilot manifest; `vp_yolo11_postprocess_test` passes; no model directory | Same-video YOLOv8/YOLO11 recall, false boxes, p95 latency, FPS, and bounded queue with no recall regression | Set `VP_WITH_YOLO11=OFF`; retain old detector |
| TensorRT pilot | `BLOCKED / NOT TESTED` | Windows TensorRT 10.7.0.23 and CUDA 12.1/13.1 inventoried; WSL SDK absent; no engine/model run | Qualify one documented CUDA/TRT pair, build external YOLO11 engine, compare outputs to ONNX, and measure real clips | Do not overwrite the old YOLOv8 engine/builder; disable TRT candidate profile |
| PaddleOCR 3.x pilot | `BLOCKED / NOT TESTED` | Paddle Inference and crop dataset absent | Same-crop recognition and latency comparison through `vp_ocr_job -> vp_ocr_consensus`, bounded queue, evidence retained on failure | Keep PaddleOCR 2.6-derived implementation and synchronous path available |

## Unexecuted Acceptance Gates

The following were not executed and must not be interpreted as passing:

- Real YOLOv8/YOLO11 ONNX inference or TensorRT engine generation.
- Real race clips, model weights, TensorRT engines, or crop datasets.
- Athlete recall, false-event, duplicate-event, OCR, latency, and FPS A/B metrics.
- Camera RTSP runtime, hardware decode, recording, reconnect recovery, or the
  30-minute reconnect/endurance test.
- PaddleOCR 3.x inference or comparison with the 2.6-derived fallback.

## Executable Profile Selection

`cycling_upgrade_sample` now has a runtime selection layer independent of the
CMake build-profile label:

| Executable profile | Detector | Tracker | OCR |
| --- | --- | --- | --- |
| `baseline` | YOLOv8 | SORT | synchronous PaddleOCR |
| `candidate` | YOLO11 | ByteTrack | asynchronous PaddleOCR |
| no `--profile` | YOLOv8 | none | none |

The no-profile behavior preserves the previous detection-only sample. Explicit
`--detector`, `--tracker`, and `--ocr` values override profile defaults
regardless of argument order. `--runtime-dir` selects model/OCR artifacts
without modifying either directory, and `--print-config` reports the effective
selection before model loading.

The CMake value `VP_RUNTIME_PROFILE=yolo11-pilot` remains an isolated build
manifest label. It must not be confused with the executable's `candidate`
profile, and it does not enable optional dependencies automatically.

Implementation validation completed on 2026-08-07:

- Standard-library option parser test passes for legacy defaults, both
  profiles, order-independent overrides, print-only mode, and invalid values.
- `cycling_upgrade_sample.cpp` passes a WSL `g++ -fsyntax-only` check with the
  optional detector/OCR macros disabled.
- Real profile startup remains `BLOCKED / NOT TESTED` because model artifacts,
  TensorRT/Paddle on one qualified build target, and race clips are absent.

See `docs/cycling-field-checklist.md` for the runtime directory layout,
camera-purchase hold points, A/B commands, promotion gates, and rollback
procedure.
