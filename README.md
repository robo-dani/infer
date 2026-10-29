# infer

TensorRT 10 inference library for YOLO models, built for the zboat segmentation server
(boat_perception, which uses it as the `third_party/infer` submodule). Its main path is YOLO26
instance segmentation on a Jetson: GPU undistortion and letterbox in one pass, per-class coverage
masks rendered straight at the caller's resolution, and the TensorRT enqueue replayed as a CUDA
graph.

Forked from [shouxieai/infer](https://github.com/shouxieai/infer).

## Build

Requires CUDA, TensorRT 10 (`nvinfer`, `nvinfer_plugin`, `nvonnxparser`), OpenCV and CMake 3.28+.
The linker path is set up for a Jetson (`/usr/lib/aarch64-linux-gnu`).

```bash
cmake -B build && cmake --build build -j
```

This produces `lib/libinfer.so` and the test tools in `build/tests/`.

## Engines

Build the engine on the device that runs it; engines do not carry across GPUs or TensorRT
versions:

```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

The I/O tensors are found by name and must be float32: input `images` (N x 3 x H x W), output
`output0` (detections) and, for segmentation, `output1` (mask prototypes, at most 32).

| `yolo::Type` | `output0` row | Post-processing |
|---|---|---|
| `V26Seg` | `x1 y1 x2 y2 conf class` + mask coefficients | threshold only (end-to-end head) |
| `V8` | `cx cy w h` + class scores | threshold + NMS |
| `V5`, `V3`, `V7`, `X` | `cx cy w h objectness` + class scores | threshold + NMS |

`output0` is N x rows x row length. Ultralytics exports YOLOv8 as N x row length x rows, so
transpose it before building a `V8` engine. Inputs are letterboxed (scaled to fit, centered, padded
with 114), converted to RGB and scaled by 1/255, except YOLOX, which takes raw BGR values.

## Usage

```c++
#include "yolo.hpp"

auto model = yolo::load("model.engine", yolo::Type::V26Seg, /*confidence*/ 0.25f);
if (!model) { /* load failed */ }

// optional: render Detections::class_mask on a mask_w x mask_h grid
model->set_class_mask_size(mask_w, mask_h);
// optional: undistort every frame on the GPU; map is the CV_32FC2 output of
// cv::initUndistortRectifyMap
model->set_source_map(map.cols, map.rows, map.ptr<float>());

yolo::Detections det;  // reuse across frames to keep its buffers
if (!model->forward(yolo::Image(frame.data, frame.cols, frame.rows), det)) {
  // inference failed; an empty det.boxes after true means nothing was detected
}
for (const yolo::Box &box : det.boxes) {
  // box.left / top / right / bottom, box.confidence, box.class_label,
  // box.seg: the instance mask at prototype resolution, null when the box is too small for one
}
// bit c of det.class_mask.data[y * mask_w + x] is set when class c covers that cell
```

`include/yolo.hpp` is the reference for the exact semantics. In short:

- Coordinates are pixel edges (pixel i spans [i, i + 1)) in the undistorted frame when a source
  map is set. Boxes from the end-to-end head are not clipped to the image.
- With a source map, inference sees exactly `cv::remap(frame, map, INTER_LINEAR,
  BORDER_REPLICATE)`, bit for bit.
- Frames in page-locked or device memory are read in place; anything else is copied to the
  device first.
- A model is not thread-safe. Without a stream argument it runs on a stream it owns, and
  `forward` returns once the results are on the host.
- `forwards` runs a batch: up to the engine's batch for a static engine, any size within the
  profile for a dynamic one.

`include/infer.hpp` is the raw TensorRT layer underneath (`trt::load`, `setTensorAddress`,
`forward(stream)`). On a non-null stream, `forward` captures `enqueueV3` into a CUDA graph on the
second call and replays it from then on; changing a tensor address or an input shape drops the
graph.

`include/cpm.hpp` wraps a model in a producer-consumer queue: `commit` returns a future, and a
worker thread batches pending inputs into `forwards`.

## Tests

Both tools read the camera calibration (`gmsl_camera_matrix`, `gmsl_dist_coeffs`) from a
`cv::FileStorage` YAML such as boat_perception's `config/default.yaml`.

```bash
# Correctness on real frames: fused undistort+letterbox vs cv::remap, boxes and masks vs a
# double-precision CPU reference, forward(undistorted) vs forward(raw) with a source map.
# --labels compares the water mask against YOLO polygon labels instead.
build/tests/check_v26seg <engine> <calibration.yaml> <raw frame dir> \
    [--undist <png dir>] [--labels <label dir>] [--max <frames>]

# Latency of forward() per frame source, TensorRT alone and the preprocess variants
build/tests/bench_v26seg <engine> <calibration.yaml> <raw frame> [iterations]
```

## Layout

- `src/infer.cu`: TensorRT engine and context, CUDA graph replay
- `src/preprocess.cu`: letterbox, undistortion, normalization
- `src/segment.cu`: instance and class mask kernels
- `src/yolo.cu`: decoding, NMS, `yolo::Infer`

`format.sh` runs clang-format over `src/`.
