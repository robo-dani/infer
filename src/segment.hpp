#ifndef __SEGMENT_HPP__
#define __SEGMENT_HPP__

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace yolo {

// One detection of a segmentation model, as the mask kernels see it
struct SegInstance {
  const float *coefficients;       // mask coefficients, one per prototype (device)
  const float *proto;              // prototype planes of its image (device)
  float left, top, right, bottom;  // box, image pixel-edge coordinates
  // image pixel-edge coordinates to prototype pixel-center ones: px = scale_x * x + offset_x
  float proto_scale_x, proto_offset_x, proto_scale_y, proto_offset_y;
  int class_label;
  int mask_width, mask_height;  // its per-instance map, 0 x 0 when the box is too small for one
  size_t mask_offset;           // where that map goes in the output buffer
};

// Prototype count the kernels support; YOLO segmentation heads have 32
const int kMaxProtoChannels = 32;

struct ProtoShape {
  int channels, width, height;
};

// logits[i] = instances[i].coefficients . prototypes: one width x height plane per instance
void compute_instance_logits(const SegInstance *instances, int count, ProtoShape proto,
                             float *logits, cudaStream_t stream);

// Per-instance maps, written to output + mask_offset. Pixel (i, j) is sigmoid(logit) * 255 at the
// image point (left + (i + 0.5) * box_width / mask_width,
//              top + (j + 0.5) * box_height / mask_height).
void render_instance_masks(const SegInstance *instances, int count, int max_mask_width,
                           int max_mask_height, const float *logits, ProtoShape proto,
                           uint8_t *output, cudaStream_t stream);

// Class-coverage mask of one image over a width x height grid spanning it. Cell (x, y) samples the
// image point ((x + 0.5) * image_width / width, (y + 0.5) * image_height / height) and gets bit c
// for every instance of class c < 8 whose box contains the point and whose logit there is > 0.
void render_class_mask(const SegInstance *instances, int count, const float *logits,
                       ProtoShape proto, int image_width, int image_height, uint8_t *mask,
                       int width, int height, cudaStream_t stream);

}  // namespace yolo

#endif  // __SEGMENT_HPP__
