#include "check.hpp"
#include "segment.hpp"

namespace yolo {

static __global__ void instance_logits_kernel(const SegInstance *instances, int count,
                                              ProtoShape proto, float *logits) {
  int area = proto.width * proto.height;
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= area) return;

  // The prototypes of an image are shared by all its instances, and reading them dominates: keep
  // this pixel's values in registers while the instances of one image go by.
  float values[kMaxProtoChannels];
  const float *cached = nullptr;
  for (int i = 0; i < count; ++i) {
    const SegInstance &instance = instances[i];
    if (instance.proto != cached) {
      cached = instance.proto;
#pragma unroll
      for (int ic = 0; ic < kMaxProtoChannels; ++ic)
        if (ic < proto.channels) values[ic] = cached[ic * area + position];
    }
    float sum = 0;
#pragma unroll
    for (int ic = 0; ic < kMaxProtoChannels; ++ic)
      if (ic < proto.channels) sum += instance.coefficients[ic] * values[ic];
    logits[(size_t)i * area + position] = sum;
  }
}

// One axis of a bilinear sample at a pixel-center coordinate, clamped to the edge as
// F.interpolate(mode="bilinear", align_corners=False) does when it upsamples prototypes
struct Tap {
  int i0, i1;
  float t;
};

static __device__ Tap tap(float x, int size) {
  x = fminf(fmaxf(x, 0.f), size - 1.f);
  int i0 = x;
  return {i0, min(i0 + 1, size - 1), x - i0};
}

static __device__ float sample_logit(const float *plane, int width, Tap tx, Tap ty) {
  const float *row0 = plane + ty.i0 * width;
  const float *row1 = plane + ty.i1 * width;
  return (row0[tx.i0] * (1 - tx.t) + row0[tx.i1] * tx.t) * (1 - ty.t) +
         (row1[tx.i0] * (1 - tx.t) + row1[tx.i1] * tx.t) * ty.t;
}

static __global__ void instance_masks_kernel(const SegInstance *instances, const float *logits,
                                             ProtoShape proto, uint8_t *output) {
  const SegInstance &instance = instances[blockIdx.z];
  int dx = blockDim.x * blockIdx.x + threadIdx.x;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= instance.mask_width || dy >= instance.mask_height) return;

  float x = instance.left + (dx + 0.5f) * (instance.right - instance.left) / instance.mask_width;
  float y = instance.top + (dy + 0.5f) * (instance.bottom - instance.top) / instance.mask_height;
  float logit =
      sample_logit(logits + (size_t)blockIdx.z * proto.width * proto.height, proto.width,
                   tap(instance.proto_scale_x * x + instance.proto_offset_x, proto.width),
                   tap(instance.proto_scale_y * y + instance.proto_offset_y, proto.height));
  float alpha = 1.0f / (1.0f + expf(-logit));
  output[instance.mask_offset + dy * instance.mask_width + dx] = alpha * 255;
}

// cells per thread, along a row: they share the instance loop and the vertical taps
static const int kClassMaskRun = 4;

static __global__ void class_mask_kernel(const SegInstance *instances, int count,
                                         const float *logits, ProtoShape proto, float scale_x,
                                         float scale_y, uint8_t *mask, int width, int height) {
  int dx = (blockDim.x * blockIdx.x + threadIdx.x) * kClassMaskRun;
  int dy = blockDim.y * blockIdx.y + threadIdx.y;
  if (dx >= width || dy >= height) return;

  float y = (dy + 0.5f) * scale_y;
  uint8_t bits[kClassMaskRun] = {};
  for (int i = 0; i < count; ++i) {
    const SegInstance &instance = instances[i];
    if (instance.class_label < 0 || instance.class_label >= 8) continue;
    if (y < instance.top || y >= instance.bottom) continue;
    const float *plane = logits + (size_t)i * proto.width * proto.height;
    Tap ty = tap(instance.proto_scale_y * y + instance.proto_offset_y, proto.height);
#pragma unroll
    for (int k = 0; k < kClassMaskRun; ++k) {
      float x = (dx + k + 0.5f) * scale_x;
      if (x < instance.left || x >= instance.right) continue;
      Tap tx = tap(instance.proto_scale_x * x + instance.proto_offset_x, proto.width);
      if (sample_logit(plane, proto.width, tx, ty) > 0) bits[k] |= 1 << instance.class_label;
    }
  }
#pragma unroll
  for (int k = 0; k < kClassMaskRun; ++k)
    if (dx + k < width) mask[dy * width + dx + k] = bits[k];
}

void compute_instance_logits(const SegInstance *instances, int count, ProtoShape proto,
                             float *logits, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((proto.width * proto.height + block.x - 1) / block.x);
  checkKernel(instance_logits_kernel<<<grid, block, 0, stream>>>(instances, count, proto, logits));
}

void render_instance_masks(const SegInstance *instances, int count, int max_mask_width,
                           int max_mask_height, const float *logits, ProtoShape proto,
                           uint8_t *output, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((max_mask_width + 31) / 32, (max_mask_height + 7) / 8, count);
  checkKernel(instance_masks_kernel<<<grid, block, 0, stream>>>(instances, logits, proto, output));
}

void render_class_mask(const SegInstance *instances, int count, const float *logits,
                       ProtoShape proto, int image_width, int image_height, uint8_t *mask,
                       int width, int height, cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((width + 32 * kClassMaskRun - 1) / (32 * kClassMaskRun), (height + 7) / 8);
  checkKernel(class_mask_kernel<<<grid, block, 0, stream>>>(
      instances, count, logits, proto, image_width / (float)width, image_height / (float)height,
      mask, width, height));
}

}  // namespace yolo
