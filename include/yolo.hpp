#ifndef __YOLO_HPP__
#define __YOLO_HPP__

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

enum class Type : int {
  V5 = 0,
  X = 1,
  V3 = 2,
  V7 = 3,
  V8 = 5,
  V26Seg = 7  // yolo26 end-to-end instance segmentation (NMS-free head)
};

// Mask of one instance at prototype resolution, covering its box. Pixel (i, j) holds
// sigmoid(logit) * 255 sampled at the image point
//   (left + (i + 0.5) * (right - left) / width, top + (j + 0.5) * (bottom - top) / height),
// which is what resizing the map to the box size and pasting it at (left, top) assumes.
struct InstanceSegmentMap {
  int width = 0, height = 0;
  unsigned char *data = nullptr;  // width * height bytes

  InstanceSegmentMap(int width, int height);
  InstanceSegmentMap(const InstanceSegmentMap &) = delete;
  InstanceSegmentMap &operator=(const InstanceSegmentMap &) = delete;
  virtual ~InstanceSegmentMap();
};

// Coordinates are pixel edges: pixel i spans [i, i + 1), so a box covering a whole W-wide image
// runs from 0 to W.
struct Box {
  float left, top, right, bottom, confidence;
  int class_label;
  std::shared_ptr<InstanceSegmentMap> seg;  // valid only in segment task

  Box() = default;
  Box(float left, float top, float right, float bottom, float confidence, int class_label)
      : left(left),
        top(top),
        right(right),
        bottom(bottom),
        confidence(confidence),
        class_label(class_label) {}
};

// BGR, 8 bits per channel, rows packed (stride width * 3). When bgrptr points into page-locked
// (cudaMallocHost, cudaHostRegister) or device memory the GPU reads the frame in place; otherwise
// it is copied to the device first.
struct Image {
  const void *bgrptr = nullptr;
  int width = 0, height = 0;

  Image() = default;
  Image(const void *bgrptr, int width, int height) : bgrptr(bgrptr), width(width), height(height) {}
};

typedef std::vector<Box> BoxArray;

// Class coverage of an image on a grid of width x height cells spanning it (row-major). Cell
// (x, y) samples the image point X = (x + 0.5) * W / width, Y = (y + 0.5) * H / height and has
// bit c set when an instance of class c covers that point: left <= X < right, top <= Y < bottom,
// and the instance's prototype logit, bilinearly interpolated there, is > 0 (sigmoid > 0.5).
// Classes >= 8 have no bit.
struct ClassMask {
  int width = 0, height = 0;
  std::vector<uint8_t> data;
};

// Result of one image. Reusing the same object across frames keeps its buffers allocated.
struct Detections {
  BoxArray boxes;
  ClassMask class_mask;  // empty unless set_class_mask_size() enabled it
};

class Infer {
 public:
  virtual ~Infer() = default;

  // Returns false when inference failed: an unusable image, CUDA/TensorRT errors. An empty
  // out.boxes after true means nothing was detected. Not thread-safe; with stream == nullptr the
  // work runs on a stream the model owns. Returns once the results are on the host.
  [[nodiscard]] virtual bool forward(const Image &image, Detections &out,
                                     void *stream = nullptr) = 0;
  // Batch version: out gets one entry per image.
  [[nodiscard]] virtual bool forwards(const std::vector<Image> &images,
                                      std::vector<Detections> &out, void *stream = nullptr) = 0;

  // Segmentation types only: from now on every forward renders Detections::class_mask on a
  // width x height grid. (0, 0) turns it off.
  virtual bool set_class_mask_size(int width, int height) = 0;

  // Undistorts every frame before inference. map_xy holds width x height (x, y) float pairs,
  // row-major: for each pixel of the undistorted image, the position it samples in the frame given
  // to forward (the CV_32FC2 output of cv::initUndistortRectifyMap). Inference then sees exactly
  // cv::remap(frame, map, INTER_LINEAR, BORDER_REPLICATE), and boxes and masks are in undistorted
  // coordinates. A CV_32FC2 map made with cv::convertMaps from the CV_16SC2 pair reproduces
  // remap with that pair bit for bit. nullptr turns undistortion off.
  virtual bool set_source_map(int width, int height, const float *map_xy) = 0;
};

std::shared_ptr<Infer> load(const std::string &engine_file, Type type,
                            float confidence_threshold = 0.25f, float nms_threshold = 0.5f);

const char *type_name(Type type);
};  // namespace yolo

#endif  // __YOLO_HPP__