#include <algorithm>
#include <cstring>

#include "check.hpp"
#include "preprocess.hpp"

namespace yolo {

Norm Norm::mean_std(const float mean[3], const float std[3], float alpha,
                    ChannelType channel_type) {
  Norm out;
  out.type = NormType::MeanStd;
  out.alpha = alpha;
  out.channel_type = channel_type;
  memcpy(out.mean, mean, sizeof(out.mean));
  memcpy(out.std, std, sizeof(out.std));
  return out;
}

Norm Norm::alpha_beta(float alpha, float beta, ChannelType channel_type) {
  Norm out;
  out.type = NormType::AlphaBeta;
  out.alpha = alpha;
  out.beta = beta;
  out.channel_type = channel_type;
  return out;
}

Norm Norm::None() { return Norm(); }

void AffineMatrix::compute(int image_width, int image_height, int network_width,
                           int network_height) {
  // Image pixels per network pixel. Taken as the larger ratio rather than inverting the float
  // scale, it is exact for integer ratios, so the sampling grid stays on whole pixels (1920x1080
  // into 640x640 reads every third pixel with bilinear weights of exactly 1 and 0).
  double inv = std::max(image_width / (double)network_width, image_height / (double)network_height);
  double pad_x = (network_width - image_width / inv) * 0.5;
  double pad_y = (network_height - image_height / inv) * 0.5;
  i2d = {{float(1 / inv), 0, float(pad_x), 0, float(1 / inv), float(pad_y)}};
  d2i = {{float(inv), 0, float(-pad_x * inv), 0, float(inv), float(-pad_y * inv)}};
}

// Undistorted pixel at source position pos, computed the way cv::remap(INTER_LINEAR,
// BORDER_REPLICATE) does for 8-bit images: the position is rounded to 1/32 px and the four
// neighbours are blended with integer weights summing to 1024, then rounded.
static __device__ void remap_pixel(const uint8_t *src, int line_size, int pixel_size, int width,
                                   int height, float2 pos, float out[3]) {
  int ix = __float2int_rn(pos.x * 32.f);
  int iy = __float2int_rn(pos.y * 32.f);
  int fx = ix & 31, fy = iy & 31;
  int x0 = ix >> 5, y0 = iy >> 5;
  int xa = min(max(x0, 0), width - 1), xb = min(max(x0 + 1, 0), width - 1);
  int ya = min(max(y0, 0), height - 1), yb = min(max(y0 + 1, 0), height - 1);
  const uint8_t *p00 = src + ya * line_size + xa * pixel_size;
  const uint8_t *p01 = src + ya * line_size + xb * pixel_size;
  const uint8_t *p10 = src + yb * line_size + xa * pixel_size;
  const uint8_t *p11 = src + yb * line_size + xb * pixel_size;
  int w00 = (32 - fx) * (32 - fy), w01 = fx * (32 - fy), w10 = (32 - fx) * fy, w11 = fx * fy;
  for (int c = 0; c < 3; ++c)
    out[c] = (p00[c] * w00 + p01[c] * w01 + p10[c] * w10 + p11[c] * w11 + 512) >> 10;
}

static __global__ void warp_affine_bilinear_and_normalize_plane_kernel(
    const uint8_t *src, int src_line_size, int src_pixel_size, int src_width, int src_height,
    SourceMap source_map, float *dst, int dst_width, int dst_height, uint8_t const_value,
    Matrix23 d2i, Norm norm) {
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= dst_width || dy >= dst_height) return;

  // the image being letterboxed: src itself or its undistortion
  int image_width = source_map.map ? source_map.width : src_width;
  int image_height = source_map.map ? source_map.height : src_height;

  // center of network pixel (dx, dy), in image pixel-center coordinates
  float src_x = d2i.v[0] * (dx + 0.5f) + d2i.v[1] * (dy + 0.5f) + d2i.v[2] - 0.5f;
  float src_y = d2i.v[3] * (dx + 0.5f) + d2i.v[4] * (dy + 0.5f) + d2i.v[5] - 0.5f;
  float c0 = const_value, c1 = const_value, c2 = const_value;

  if (!(src_x <= -1 || src_x >= image_width || src_y <= -1 || src_y >= image_height)) {
    int x_low = floorf(src_x);
    int y_low = floorf(src_y);
    float lx = src_x - x_low;
    float ly = src_y - y_low;
    float hx = 1 - lx;
    float hy = 1 - ly;
    float weights[4] = {hy * hx, hy * lx, ly * hx, ly * lx};
    float sum[3] = {0, 0, 0};
    for (int k = 0; k < 4; ++k) {
      // zero weights are the norm at integer scale ratios and save the map lookup
      if (weights[k] == 0) continue;
      int x = x_low + (k & 1), y = y_low + (k >> 1);
      float v[3] = {(float)const_value, (float)const_value, (float)const_value};
      if (x >= 0 && x < image_width && y >= 0 && y < image_height) {
        if (source_map.map) {
          remap_pixel(src, src_line_size, src_pixel_size, src_width, src_height,
                      source_map.map[y * source_map.width + x], v);
        } else {
          const uint8_t *p = src + y * src_line_size + x * src_pixel_size;
          v[0] = p[0];
          v[1] = p[1];
          v[2] = p[2];
        }
      }
      sum[0] += weights[k] * v[0];
      sum[1] += weights[k] * v[1];
      sum[2] += weights[k] * v[2];
    }

    // same to opencv
    c0 = floorf(sum[0] + 0.5f);
    c1 = floorf(sum[1] + 0.5f);
    c2 = floorf(sum[2] + 0.5f);
  }

  if (norm.channel_type == ChannelType::SwapRB) {
    float t = c2;
    c2 = c0;
    c0 = t;
  }

  if (norm.type == NormType::MeanStd) {
    c0 = (c0 * norm.alpha - norm.mean[0]) / norm.std[0];
    c1 = (c1 * norm.alpha - norm.mean[1]) / norm.std[1];
    c2 = (c2 * norm.alpha - norm.mean[2]) / norm.std[2];
  } else if (norm.type == NormType::AlphaBeta) {
    c0 = c0 * norm.alpha + norm.beta;
    c1 = c1 * norm.alpha + norm.beta;
    c2 = c2 * norm.alpha + norm.beta;
  }

  int area = dst_width * dst_height;
  float *pdst_c0 = dst + dy * dst_width + dx;
  float *pdst_c1 = pdst_c0 + area;
  float *pdst_c2 = pdst_c1 + area;
  *pdst_c0 = c0;
  *pdst_c1 = c1;
  *pdst_c2 = c2;
}

void warp_affine_bilinear_and_normalize_plane(const uint8_t *src, int src_line_size,
                                              int src_pixel_size, int src_width, int src_height,
                                              const SourceMap &source_map, float *dst,
                                              int dst_width, int dst_height, const Matrix23 &d2i,
                                              uint8_t const_value, const Norm &norm,
                                              cudaStream_t stream) {
  dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
  dim3 block(32, 32);

  checkKernel(warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block, 0, stream>>>(
      src, src_line_size, src_pixel_size, src_width, src_height, source_map, dst, dst_width,
      dst_height, const_value, d2i, norm));
}

}  // namespace yolo
