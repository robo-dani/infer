// Latency of the V26Seg path for the frame sources a caller can have.
//
//   bench_v26seg <engine> <calibration.yaml> <raw frame> [iterations]
//
// U is the raw frame undistorted on the CPU with cv::remap (CV_16SC2 maps), the way a caller does
// without set_source_map. Pinned means the frame sits in cudaMallocHost memory. BGRx is the raw
// frame with a 4th byte per pixel, as a camera pipeline without a BGRx-to-BGR conversion gives.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <string>

#include "infer.hpp"
#include "preprocess.hpp"
#include "yolo.hpp"

#define CHECK_CUDA(call)                                                             \
  do {                                                                               \
    cudaError_t e = (call);                                                          \
    if (e != cudaSuccess) {                                                          \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1);                                                                       \
    }                                                                                \
  } while (0)

using clk = std::chrono::steady_clock;

template <class F>
static double average_ms(int n, F f) {
  for (int i = 0; i < 20; ++i) f();
  auto t = clk::now();
  for (int i = 0; i < n; ++i) f();
  return std::chrono::duration<double, std::milli>(clk::now() - t).count() / n;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("usage: %s <engine> <calibration.yaml> <raw frame> [iterations]\n", argv[0]);
    return 1;
  }
  int n = argc > 4 ? atoi(argv[4]) : 200;

  cv::FileStorage storage(argv[2], cv::FileStorage::READ);
  cv::Mat K, D;
  storage["gmsl_camera_matrix"] >> K;
  storage["gmsl_dist_coeffs"] >> D;
  cv::Mat raw = cv::imread(argv[3]), map1, map2, map_xy, undist;
  if (raw.empty() || K.empty() || D.empty()) return 1;
  cv::initUndistortRectifyMap(K, D, cv::noArray(), K, raw.size(), CV_16SC2, map1, map2);
  cv::convertMaps(map1, map2, map_xy, cv::noArray(), CV_32FC2);
  cv::remap(raw, undist, map1, map2, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  cv::Mat raw_bgrx;
  cv::cvtColor(raw, raw_bgrx, cv::COLOR_BGR2BGRA);

  size_t bytes = raw.total() * 3;
  uint8_t *raw_pinned, *undist_pinned;
  CHECK_CUDA(cudaMallocHost(&raw_pinned, bytes));
  CHECK_CUDA(cudaMallocHost(&undist_pinned, bytes));
  memcpy(raw_pinned, raw.data, bytes);
  memcpy(undist_pinned, undist.data, bytes);
  int w = raw.cols, h = raw.rows;

  auto model = yolo::load(argv[1], yolo::Type::V26Seg, 0.4f);
  if (!model) return 1;
  yolo::Detections out;
  auto forward = [&](const void *frame, int channels = 3) {
    if (!model->forward(yolo::Image(frame, w, h, channels), out)) exit(1);
  };

  printf("frame %dx%d, %d iterations\n\n", w, h, n);
  printf("cv::remap CV_16SC2 (CPU)                 %7.3f ms\n", average_ms(n / 4 + 1, [&] {
           cv::Mat o;
           cv::remap(raw, o, map1, map2, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
         }));

  model->set_class_mask_size(0, 0);
  printf("forward U pageable, no class mask        %7.3f ms\n",
         average_ms(n, [&] { forward(undist.data); }));
  model->set_class_mask_size(960, 540);
  printf("forward U pageable, class mask 960x540   %7.3f ms  (%zu boxes)\n",
         average_ms(n, [&] { forward(undist.data); }), out.boxes.size());
  printf("forward U pinned                         %7.3f ms\n",
         average_ms(n, [&] { forward(undist_pinned); }));
  model->set_source_map(w, h, map_xy.ptr<float>());
  printf("forward raw pageable + source map        %7.3f ms\n",
         average_ms(n, [&] { forward(raw.data); }));
  printf("forward raw pinned + source map          %7.3f ms\n",
         average_ms(n, [&] { forward(raw_pinned); }));
  printf("forward raw BGRx pageable + source map   %7.3f ms\n",
         average_ms(n, [&] { forward(raw_bgrx.data, 4); }));
  model->set_source_map(0, 0, nullptr);

  // TensorRT alone, on a context of its own (forward replays a CUDA graph)
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto trt = trt::load(argv[1]);
  float *tensor = nullptr;
  for (auto name : {"images", "output0", "output1"}) {
    size_t numel = 1;
    for (int d : trt->static_dims(name)) numel *= d;
    float *p;
    CHECK_CUDA(cudaMalloc(&p, numel * sizeof(float)));
    trt->setTensorAddress(name, p);
    if (!tensor) tensor = p;
  }
  printf("\nTensorRT enqueue + sync                  %7.3f ms\n", average_ms(n, [&] {
           if (!trt->forward(stream)) exit(1);
           CHECK_CUDA(cudaStreamSynchronize(stream));
         }));

  // preprocess alone, through the library's kernel
  int net_w = trt->static_dims("images")[3], net_h = trt->static_dims("images")[2];
  uint8_t *frame_device;
  float *map_device;
  CHECK_CUDA(cudaMalloc(&frame_device, raw_bgrx.total() * 4));
  CHECK_CUDA(cudaMalloc(&map_device, map_xy.total() * 2 * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(map_device, map_xy.ptr<float>(), map_xy.total() * 2 * sizeof(float),
                        cudaMemcpyHostToDevice));
  yolo::AffineMatrix affine;
  affine.compute(w, h, net_w, net_h);
  auto norm = yolo::Norm::alpha_beta(1 / 255.0f, 0.0f, yolo::ChannelType::SwapRB);
  yolo::SourceMap no_map, with_map{(const float2 *)map_device, w, h};
  auto letterbox = [&](const uint8_t *src, const yolo::SourceMap &map, int channels = 3) {
    yolo::warp_affine_bilinear_and_normalize_plane(src, w * channels, channels, w, h, map, tensor,
                                                   net_w, net_h, affine.d2i, 114, norm, stream);
  };
  printf("preprocess only (letterbox to %dx%d):\n", net_w, net_h);
  printf("  H2D pageable + letterbox               %7.3f ms\n", average_ms(n, [&] {
           CHECK_CUDA(
               cudaMemcpyAsync(frame_device, undist.data, bytes, cudaMemcpyHostToDevice, stream));
           letterbox(frame_device, no_map);
           CHECK_CUDA(cudaStreamSynchronize(stream));
         }));
  printf("  letterbox from pinned                  %7.3f ms\n", average_ms(n, [&] {
           letterbox(undist_pinned, no_map);
           CHECK_CUDA(cudaStreamSynchronize(stream));
         }));
  printf(
      "  H2D pageable + remap letterbox         %7.3f ms\n", average_ms(n, [&] {
        CHECK_CUDA(cudaMemcpyAsync(frame_device, raw.data, bytes, cudaMemcpyHostToDevice, stream));
        letterbox(frame_device, with_map);
        CHECK_CUDA(cudaStreamSynchronize(stream));
      }));
  printf("  remap letterbox from pinned            %7.3f ms\n", average_ms(n, [&] {
           letterbox(raw_pinned, with_map);
           CHECK_CUDA(cudaStreamSynchronize(stream));
         }));
  printf("  remap letterbox from device            %7.3f ms\n", average_ms(n, [&] {
           letterbox(frame_device, with_map);
           CHECK_CUDA(cudaStreamSynchronize(stream));
         }));
  CHECK_CUDA(cudaMemcpy(frame_device, raw_bgrx.data, raw_bgrx.total() * 4, cudaMemcpyHostToDevice));
  printf("  remap letterbox from device, BGRx      %7.3f ms\n", average_ms(n, [&] {
           letterbox(frame_device, with_map, 4);
           CHECK_CUDA(cudaStreamSynchronize(stream));
         }));
  return 0;
}
