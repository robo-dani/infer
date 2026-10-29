#ifndef __PREPROCESS_HPP__
#define __PREPROCESS_HPP__

#include <cuda_runtime.h>

#include <cstdint>

namespace yolo {

enum class NormType : int { None = 0, MeanStd = 1, AlphaBeta = 2 };

enum class ChannelType : int { None = 0, SwapRB = 1 };

// Normalization: mean/std or alpha/beta, optionally swapping R and B
struct Norm {
  float mean[3];
  float std[3];
  float alpha, beta;
  NormType type = NormType::None;
  ChannelType channel_type = ChannelType::None;

  // out = (x * alpha - mean) / std
  static Norm mean_std(const float mean[3], const float std[3], float alpha = 1 / 255.0f,
                       ChannelType channel_type = ChannelType::None);

  // out = x * alpha + beta
  static Norm alpha_beta(float alpha, float beta = 0, ChannelType channel_type = ChannelType::None);

  // None
  static Norm None();
};

// 2x3 affine matrix, passed to kernels by value
struct Matrix23 {
  float v[6];
};

static __host__ __device__ inline void affine_project(const Matrix23 &m, float x, float y,
                                                      float *ox, float *oy) {
  *ox = m.v[0] * x + m.v[1] * y + m.v[2];
  *oy = m.v[3] * x + m.v[4] * y + m.v[5];
}

// Letterbox transform between an image and the network input: scale to fit, then center. Both
// matrices map pixel-edge coordinates, where pixel i spans [i, i + 1): the convention of boxes and
// masks. Kernels that sample at pixel centers add the 0.5 themselves.
struct AffineMatrix {
  Matrix23 i2d;  // image to network
  Matrix23 d2i;  // network to image

  void compute(int image_width, int image_height, int network_width, int network_height);
};

// Undistortion in front of the letterbox. map[y * width + x] is the position in the source frame
// that undistorted pixel (x, y) samples: the CV_32FC2 output of cv::initUndistortRectifyMap.
struct SourceMap {
  const float2 *map = nullptr;  // device memory; nullptr means no undistortion
  int width = 0, height = 0;    // undistorted image size
};

// Letterboxes the image into dst (planar float, dst_width x dst_height) with bilinear sampling,
// const_value outside the image and then norm. The image is src (device-accessible; BGR in the
// first 3 of every src_pixel_size bytes: 3 for BGR, 4 for BGRx) or, with a source map,
// cv::remap(src, map, INTER_LINEAR, BORDER_REPLICATE) evaluated on the fly with OpenCV's
// fixed-point arithmetic, so both give the same tensor bit for bit.
void warp_affine_bilinear_and_normalize_plane(const uint8_t *src, int src_line_size,
                                              int src_pixel_size, int src_width, int src_height,
                                              const SourceMap &source_map, float *dst,
                                              int dst_width, int dst_height, const Matrix23 &d2i,
                                              uint8_t const_value, const Norm &norm,
                                              cudaStream_t stream);

}  // namespace yolo

#endif  // __PREPROCESS_HPP__
