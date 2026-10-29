// Correctness checks of the V26Seg path on real frames, against OpenCV and against an independent
// double-precision CPU implementation of the Ultralytics mask semantics.
//
//   check_v26seg <engine> <calibration.yaml> <raw frame dir> [--undist <png dir>]
//                [--labels <yolo label dir>] [--max <frames>]
//
// U is the raw frame undistorted with cv::remap and the CV_16SC2 maps, as production does. Per
// frame:
//   tensor  letterboxing the raw frame through a source map vs letterboxing U, element-wise
//   ref     library boxes and class masks vs the CPU reference fed with TensorRT outputs for the
//           same input tensor, plus the water mask of the previous library (7ec96f5 and the
//           caller's resize-and-paste post-processing), emulated
//   e2e     forward(U) vs forward(raw) with set_source_map, from pageable and page-locked memory
// With --labels the frames are taken as labelled, without undistortion: ref runs on the raw frame
// and water IoU against the label polygons is reported for the old and the new mask.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "infer.hpp"
#include "preprocess.hpp"
#include "yolo.hpp"

namespace fs = std::filesystem;
using std::string;
using std::vector;

#define CHECK_CUDA(call)                                                             \
  do {                                                                               \
    cudaError_t e = (call);                                                          \
    if (e != cudaSuccess) {                                                          \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      exit(1);                                                                       \
    }                                                                                \
  } while (0)

static const float kConfidence = 0.4f;
static const int kGridW = 960, kGridH = 540;
static const int kWater = 0;

// the library's letterbox kernel, run on a host frame
static vector<float> letterbox(const cv::Mat &src, const float *map_device, int map_w, int map_h,
                               int net_w, int net_h) {
  size_t bytes = src.total() * 3;
  uint8_t *src_device;
  float *dst_device;
  CHECK_CUDA(cudaMalloc(&src_device, bytes));
  CHECK_CUDA(cudaMalloc(&dst_device, net_w * net_h * 3 * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(src_device, src.data, bytes, cudaMemcpyHostToDevice));
  yolo::SourceMap source_map;
  if (map_device) source_map = {(const float2 *)map_device, map_w, map_h};
  yolo::AffineMatrix affine;
  affine.compute(map_device ? map_w : src.cols, map_device ? map_h : src.rows, net_w, net_h);
  yolo::warp_affine_bilinear_and_normalize_plane(
      src_device, src.cols * 3, src.cols, src.rows, source_map, dst_device, net_w, net_h,
      affine.d2i, 114, yolo::Norm::alpha_beta(1 / 255.0f, 0.0f, yolo::ChannelType::SwapRB),
      nullptr);
  CHECK_CUDA(cudaDeviceSynchronize());
  vector<float> out(net_w * net_h * 3);
  CHECK_CUDA(
      cudaMemcpy(out.data(), dst_device, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(src_device));
  CHECK_CUDA(cudaFree(dst_device));
  return out;
}

struct TensorDiff {
  long elements = 0, mismatches = 0;
  int max_diff = 0;  // in 8-bit levels
  void add(const vector<float> &a, const vector<float> &b) {
    for (size_t i = 0; i < a.size(); ++i) {
      int d = std::lround(std::fabs(a[i] - b[i]) * 255);
      mismatches += d != 0;
      max_diff = std::max(max_diff, d);
    }
    elements += a.size();
  }
};

// Raw TensorRT outputs for one input tensor, from a context of its own
struct Reference {
  std::shared_ptr<trt::Infer> trt;
  vector<int> in_dims, out0_dims, out1_dims;
  float *in_device, *out0_device, *out1_device;
  vector<float> out0, out1;

  bool load(const string &engine) {
    trt = trt::load(engine);
    if (!trt) return false;
    in_dims = trt->static_dims("images");
    out0_dims = trt->static_dims("output0");
    out1_dims = trt->static_dims("output1");
    out0.resize(out0_dims[1] * out0_dims[2]);
    out1.resize(out1_dims[1] * out1_dims[2] * out1_dims[3]);
    CHECK_CUDA(cudaMalloc(&in_device, in_dims[1] * in_dims[2] * in_dims[3] * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out0_device, out0.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out1_device, out1.size() * sizeof(float)));
    trt->setTensorAddress("images", in_device);
    trt->setTensorAddress("output0", out0_device);
    trt->setTensorAddress("output1", out1_device);
    return true;
  }

  void run(const vector<float> &tensor) {
    CHECK_CUDA(cudaMemcpy(in_device, tensor.data(), tensor.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    if (!trt->forward(nullptr)) {
      printf("reference forward failed\n");
      exit(1);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(
        cudaMemcpy(out0.data(), out0_device, out0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(out1.data(), out1_device, out1.size() * sizeof(float), cudaMemcpyDeviceToHost));
  }
};

// Ultralytics semantics in double precision: boxes are pixel-edge coordinates, letterboxed as
// x_net = x / inv + pad; prototypes are upsampled bilinearly with align_corners=False.
struct RefInstance {
  double left, top, right, bottom;
  float confidence, x1_net, y1_net, x2_net, y2_net;
  int label;
  vector<double> logits;  // prototype resolution
};

struct RefModel {
  int image_w, image_h, net_w, net_h, proto_w, proto_h, channels;
  double inv, pad_x, pad_y;
  vector<RefInstance> instances;

  RefModel(const Reference &r, int image_w, int image_h)
      : image_w(image_w),
        image_h(image_h),
        net_w(r.in_dims[3]),
        net_h(r.in_dims[2]),
        proto_w(r.out1_dims[3]),
        proto_h(r.out1_dims[2]),
        channels(r.out1_dims[1]) {
    inv = std::max(image_w / (double)net_w, image_h / (double)net_h);
    pad_x = (net_w - image_w / inv) / 2;
    pad_y = (net_h - image_h / inv) / 2;
    int cdim = r.out0_dims[2];
    for (int row = 0; row < r.out0_dims[1]; ++row) {
      const float *p = r.out0.data() + row * cdim;
      if (p[4] < kConfidence) continue;
      RefInstance in;
      in.x1_net = p[0], in.y1_net = p[1], in.x2_net = p[2], in.y2_net = p[3];
      in.left = (p[0] - pad_x) * inv;
      in.top = (p[1] - pad_y) * inv;
      in.right = (p[2] - pad_x) * inv;
      in.bottom = (p[3] - pad_y) * inv;
      in.confidence = p[4];
      in.label = p[5];
      in.logits.assign(proto_w * proto_h, 0.0);
      for (int c = 0; c < channels; ++c) {
        const float *plane = r.out1.data() + c * proto_w * proto_h;
        for (int i = 0; i < proto_w * proto_h; ++i) in.logits[i] += (double)p[6 + c] * plane[i];
      }
      instances.push_back(std::move(in));
    }
  }

  double logit(const RefInstance &in, double x, double y) const {
    double px = (x / inv + pad_x) * proto_w / net_w - 0.5;
    double py = (y / inv + pad_y) * proto_h / net_h - 0.5;
    px = std::clamp(px, 0.0, proto_w - 1.0);
    py = std::clamp(py, 0.0, proto_h - 1.0);
    int x0 = px, y0 = py;
    int x1 = std::min(x0 + 1, proto_w - 1), y1 = std::min(y0 + 1, proto_h - 1);
    double lx = px - x0, ly = py - y0;
    const double *l = in.logits.data();
    return (l[y0 * proto_w + x0] * (1 - lx) + l[y0 * proto_w + x1] * lx) * (1 - ly) +
           (l[y1 * proto_w + x0] * (1 - lx) + l[y1 * proto_w + x1] * lx) * ly;
  }

  cv::Mat class_mask(int w, int h) const {
    cv::Mat mask(h, w, CV_8U);
    for (int gy = 0; gy < h; ++gy) {
      for (int gx = 0; gx < w; ++gx) {
        double x = (gx + 0.5) * image_w / w, y = (gy + 0.5) * image_h / h;
        uint8_t bits = 0;
        for (auto &in : instances) {
          if (in.label < 0 || in.label >= 8) continue;
          if (x < in.left || x >= in.right || y < in.top || y >= in.bottom) continue;
          if (logit(in, x, y) > 0) bits |= 1 << in.label;
        }
        mask.at<uint8_t>(gy, gx) = bits;
      }
    }
    return mask;
  }

  // 7ec96f5: per-instance maps sampled at whole prototype pixels from the truncated origin
  // (left_net * r), boxes mapped with the pixel-center matrix (+0.5 * (inv - 1) px); the caller
  // resized each map to its box, pasted it at the rounded box origin, thresholded at 128 and
  // downsampled the full frame with INTER_NEAREST.
  cv::Mat old_water_mask(int w, int h) const {
    cv::Mat full = cv::Mat::zeros(image_h, image_w, CV_8U);
    const cv::Rect image_rect(0, 0, image_w, image_h);
    float sx = proto_w / (float)net_w, sy = proto_h / (float)net_h;
    double shift = 0.5 * (inv - 1);
    for (auto &in : instances) {
      if (in.label != kWater) continue;
      int mw = (in.x2_net - in.x1_net) * sx + 0.5f;
      int mh = (in.y2_net - in.y1_net) * sy + 0.5f;
      if (mw <= 0 || mh <= 0) continue;
      int ox = in.x1_net * sx, oy = in.y1_net * sy;  // truncated toward zero, as the int params
      cv::Mat seg(mh, mw, CV_8U);
      for (int j = 0; j < mh; ++j) {
        for (int i = 0; i < mw; ++i) {
          int px = ox + i, py = oy + j;
          uint8_t v = 0;
          if (px >= 0 && px < proto_w && py >= 0 && py < proto_h)
            v = 255 / (1 + std::exp(-in.logits[py * proto_w + px]));
          seg.at<uint8_t>(j, i) = v;
        }
      }
      double left = in.left + shift, top = in.top + shift;
      double right = in.right + shift, bottom = in.bottom + shift;
      int bw = std::max(1, (int)std::round(right - left));
      int bh = std::max(1, (int)std::round(bottom - top));
      int x = std::round(left), y = std::round(top);
      cv::Rect box = cv::Rect(x, y, bw, bh) & image_rect;
      if (box.empty()) continue;
      cv::Mat resized;
      cv::resize(seg, resized, cv::Size(bw, bh));
      full(box).setTo(255, resized(cv::Rect(box.x - x, box.y - y, box.width, box.height)) > 128);
    }
    cv::Mat out;
    cv::resize(full, out, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
    return out;
  }
};

static cv::Mat bit(const cv::Mat &mask, int b) { return (mask & (1 << b)) != 0; }
static cv::Mat to_mat(const yolo::ClassMask &m) {
  return cv::Mat(m.height, m.width, CV_8U, (void *)m.data.data()).clone();
}

static double iou(const cv::Mat &a, const cv::Mat &b) {
  double inter = cv::countNonZero(a & b), uni = cv::countNonZero(a | b);
  return uni == 0 ? 1.0 : inter / uni;
}

// largest coordinate difference between two box sets, matched greedily by distance
static double box_diff(const yolo::BoxArray &a, const vector<RefInstance> &b, int *unmatched) {
  double worst = 0;
  vector<bool> used(b.size());
  *unmatched = std::abs((int)a.size() - (int)b.size());
  for (auto &box : a) {
    int best = -1;
    double best_d = 1e30;
    for (size_t j = 0; j < b.size(); ++j) {
      if (used[j] || b[j].label != box.class_label) continue;
      double d = std::max({std::fabs(box.left - b[j].left), std::fabs(box.top - b[j].top),
                           std::fabs(box.right - b[j].right), std::fabs(box.bottom - b[j].bottom)});
      if (d < best_d) best_d = d, best = j;
    }
    if (best < 0) {
      ++*unmatched;
      continue;
    }
    used[best] = true;
    worst = std::max(worst, best_d);
  }
  return worst;
}

static double box_diff(const yolo::BoxArray &a, const yolo::BoxArray &b, int *unmatched) {
  vector<RefInstance> as_ref;
  for (auto &box : b) {
    RefInstance in;
    in.left = box.left, in.top = box.top, in.right = box.right, in.bottom = box.bottom;
    in.label = box.class_label;
    as_ref.push_back(std::move(in));
  }
  return box_diff(a, as_ref, unmatched);
}

// per-instance maps vs the reference: largest difference in 8-bit levels
static int seg_diff(const yolo::BoxArray &boxes, const RefModel &ref) {
  int worst = 0;
  for (auto &box : boxes) {
    if (!box.seg) continue;
    const RefInstance *match = nullptr;
    double best = 1e30;
    for (auto &in : ref.instances) {
      double d = std::fabs(box.left - in.left) + std::fabs(box.top - in.top);
      if (in.label == box.class_label && d < best) best = d, match = &in;
    }
    if (!match) continue;
    int mw = box.seg->width, mh = box.seg->height;
    for (int j = 0; j < mh; ++j) {
      for (int i = 0; i < mw; ++i) {
        double x = box.left + (i + 0.5) * (box.right - box.left) / mw;
        double y = box.top + (j + 0.5) * (box.bottom - box.top) / mh;
        int expect = 255 / (1 + std::exp(-ref.logit(*match, x, y)));
        worst = std::max(worst, std::abs(expect - box.seg->data[j * mw + i]));
      }
    }
  }
  return worst;
}

// YOLO polygon labels of class water, rasterized onto a w x h grid over the image
static cv::Mat label_water(const string &path, int w, int h) {
  cv::Mat mask = cv::Mat::zeros(h, w, CV_8U);
  std::ifstream in(path);
  string line;
  const int shift = 8;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    int cls;
    ss >> cls;
    if (cls != kWater) continue;
    vector<cv::Point> poly;
    double x, y;
    // label coordinates are pixel edges, fillPoly's are pixel centers
    while (ss >> x >> y)
      poly.emplace_back(std::lround((x * w - 0.5) * (1 << shift)),
                        std::lround((y * h - 0.5) * (1 << shift)));
    vector<vector<cv::Point>> polys{poly};
    cv::fillPoly(mask, polys, 255, cv::LINE_8, shift);
  }
  return mask;
}

// boat_perception's border snap (segmentor.cpp)
static void snap_water_to_borders(cv::Mat &mask, int snap_px) {
  int w = mask.cols, h = mask.rows;
  for (int y = 0; y < h; ++y) {
    uchar *row = mask.ptr<uchar>(y);
    for (int x = 1; x <= snap_px && x < w; ++x) {
      if (row[x]) {
        memset(row, 255, x);
        break;
      }
    }
    for (int x = w - 2; x >= w - 1 - snap_px && x >= 0; --x) {
      if (row[x]) {
        memset(row + x + 1, 255, w - 1 - x);
        break;
      }
    }
  }
  for (int x = 0; x < w; ++x) {
    for (int y = h - 2; y >= h - 1 - snap_px && y >= 0; --y) {
      if (mask.at<uchar>(y, x)) {
        for (int yy = y + 1; yy < h; ++yy) mask.at<uchar>(yy, x) = 255;
        break;
      }
    }
  }
}

static double median(vector<double> v) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf(
        "usage: %s <engine> <calibration.yaml> <raw frame dir> [--undist dir] [--labels dir] "
        "[--max n]\n",
        argv[0]);
    return 1;
  }
  string engine = argv[1], calibration = argv[2], raw_dir = argv[3], undist_dir, labels_dir;
  int max_frames = 1 << 30;
  for (int i = 4; i + 1 < argc; i += 2) {
    string k = argv[i];
    if (k == "--undist") undist_dir = argv[i + 1];
    if (k == "--labels") labels_dir = argv[i + 1];
    if (k == "--max") max_frames = atoi(argv[i + 1]);
  }

  cv::FileStorage storage(calibration, cv::FileStorage::READ);
  cv::Mat K, D;
  storage["gmsl_camera_matrix"] >> K;
  storage["gmsl_dist_coeffs"] >> D;
  if (K.empty() || D.empty()) {
    printf("no gmsl_camera_matrix / gmsl_dist_coeffs in %s\n", calibration.c_str());
    return 1;
  }

  vector<fs::path> frames;
  for (auto &e : fs::directory_iterator(raw_dir))
    if (e.path().extension() == ".jpg" || e.path().extension() == ".png")
      frames.push_back(e.path());
  std::sort(frames.begin(), frames.end());
  if ((int)frames.size() > max_frames) frames.resize(max_frames);

  auto model = yolo::load(engine, yolo::Type::V26Seg, kConfidence);
  Reference reference;
  if (!model || !reference.load(engine)) return 1;
  int net_w = reference.in_dims[3], net_h = reference.in_dims[2];

  cv::Size size;
  cv::Mat map1, map2, map_converted, map_float;
  float *map_converted_device = nullptr, *map_float_device = nullptr;
  uint8_t *pinned = nullptr;
  size_t pinned_bytes = 0;

  TensorDiff converted_vs_remap, float_vs_remapf, float_vs_remap;
  long undist_png_mismatch = 0, undist_png_frames = 0;
  long ref_cells = 0, ref_mismatch_small = 0, ref_mismatch_full = 0;
  long e2e_cells = 0, e2e_mismatch = 0, pinned_mismatch = 0, e2e_unmatched = 0;
  double worst_box = 0, worst_e2e_box = 0;
  int worst_seg = 0, ref_unmatched = 0;
  vector<double> iou_new, iou_old, gt_new, gt_old, gt_new_snap, gt_old_snap;

  for (auto &path : frames) {
    cv::Mat raw = cv::imread(path.string());
    if (raw.empty()) continue;
    if (raw.size() != size) {
      size = raw.size();
      cv::initUndistortRectifyMap(K, D, cv::noArray(), K, size, CV_16SC2, map1, map2);
      cv::convertMaps(map1, map2, map_converted, cv::noArray(), CV_32FC2);
      cv::initUndistortRectifyMap(K, D, cv::noArray(), K, size, CV_32FC2, map_float, cv::noArray());
      size_t map_bytes = size.area() * 2 * sizeof(float);
      CHECK_CUDA(cudaFree(map_converted_device));
      CHECK_CUDA(cudaFree(map_float_device));
      CHECK_CUDA(cudaMalloc(&map_converted_device, map_bytes));
      CHECK_CUDA(cudaMalloc(&map_float_device, map_bytes));
      CHECK_CUDA(cudaMemcpy(map_converted_device, map_converted.ptr<float>(), map_bytes,
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(
          cudaMemcpy(map_float_device, map_float.ptr<float>(), map_bytes, cudaMemcpyHostToDevice));
    }

    cv::Mat undist, undist_float;
    cv::remap(raw, undist, map1, map2, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::remap(raw, undist_float, map_float, cv::noArray(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    if (!undist_dir.empty()) {
      cv::Mat png = cv::imread((fs::path(undist_dir) / path.stem()).string() + ".png");
      if (!png.empty()) {
        cv::Mat diff;
        cv::absdiff(png, undist, diff);
        undist_png_mismatch += cv::countNonZero(diff.reshape(1));
        ++undist_png_frames;
      }
    }

    // tensor: undistortion fused into the letterbox vs cv::remap then letterbox
    auto t_remap = letterbox(undist, nullptr, 0, 0, net_w, net_h);
    converted_vs_remap.add(
        letterbox(raw, map_converted_device, size.width, size.height, net_w, net_h), t_remap);
    auto t_float = letterbox(raw, map_float_device, size.width, size.height, net_w, net_h);
    float_vs_remapf.add(t_float, letterbox(undist_float, nullptr, 0, 0, net_w, net_h));
    float_vs_remap.add(t_float, t_remap);

    // ref: the library vs the CPU reference on the same input tensor
    const cv::Mat &image = labels_dir.empty() ? undist : raw;
    reference.run(labels_dir.empty() ? t_remap : letterbox(raw, nullptr, 0, 0, net_w, net_h));
    RefModel ref(reference, image.cols, image.rows);
    yolo::Detections small, full;
    if (!model->set_class_mask_size(kGridW, kGridH) ||
        !model->forward(yolo::Image(image.data, image.cols, image.rows), small) ||
        !model->set_class_mask_size(image.cols, image.rows) ||
        !model->forward(yolo::Image(image.data, image.cols, image.rows), full)) {
      printf("forward failed on %s\n", path.c_str());
      return 1;
    }
    cv::Mat ref_small = ref.class_mask(kGridW, kGridH);
    cv::Mat ref_full = ref.class_mask(image.cols, image.rows);
    cv::Mat lib_small = to_mat(small.class_mask), lib_full = to_mat(full.class_mask);
    ref_cells += ref_small.total();
    ref_mismatch_small += cv::countNonZero(lib_small != ref_small);
    ref_mismatch_full += cv::countNonZero(lib_full != ref_full);
    int unmatched = 0;
    worst_box = std::max(worst_box, box_diff(small.boxes, ref.instances, &unmatched));
    ref_unmatched += unmatched;
    worst_seg = std::max(worst_seg, seg_diff(small.boxes, ref));
    cv::Mat water_ref = bit(ref_small, kWater), water_new = bit(lib_small, kWater);
    cv::Mat water_old = ref.old_water_mask(kGridW, kGridH);
    iou_new.push_back(iou(water_new, water_ref));
    iou_old.push_back(iou(water_old, water_ref));

    if (!labels_dir.empty()) {
      cv::Mat gt =
          label_water((fs::path(labels_dir) / path.stem()).string() + ".txt", kGridW, kGridH);
      gt_new.push_back(iou(water_new, gt));
      gt_old.push_back(iou(water_old, gt));
      // the caller snaps 8 px at full resolution, 4 px on this half-resolution grid
      snap_water_to_borders(water_new, 4);
      snap_water_to_borders(water_old, 4);
      gt_new_snap.push_back(iou(water_new, gt));
      gt_old_snap.push_back(iou(water_old, gt));
    }

    // e2e: forward(U) vs forward(raw) undistorted by the library, pageable and page-locked
    yolo::Detections via_map, via_pinned;
    if (raw.total() * 3 > pinned_bytes) {
      CHECK_CUDA(cudaFreeHost(pinned));
      pinned_bytes = raw.total() * 3;
      CHECK_CUDA(cudaMallocHost(&pinned, pinned_bytes));
    }
    memcpy(pinned, raw.data, raw.total() * 3);
    yolo::Detections plain;
    if (!model->set_class_mask_size(kGridW, kGridH) ||
        !model->forward(yolo::Image(undist.data, undist.cols, undist.rows), plain) ||
        !model->set_source_map(size.width, size.height, map_converted.ptr<float>()) ||
        !model->forward(yolo::Image(raw.data, raw.cols, raw.rows), via_map) ||
        !model->forward(yolo::Image(pinned, raw.cols, raw.rows), via_pinned) ||
        !model->set_source_map(0, 0, nullptr)) {
      printf("e2e forward failed on %s\n", path.c_str());
      return 1;
    }
    e2e_cells += plain.class_mask.data.size();
    e2e_mismatch += cv::countNonZero(to_mat(plain.class_mask) != to_mat(via_map.class_mask));
    pinned_mismatch +=
        cv::countNonZero(to_mat(via_pinned.class_mask) != to_mat(via_map.class_mask));
    worst_e2e_box = std::max(worst_e2e_box, box_diff(plain.boxes, via_map.boxes, &unmatched));
    e2e_unmatched += unmatched;
    worst_e2e_box = std::max(worst_e2e_box, box_diff(via_pinned.boxes, via_map.boxes, &unmatched));
    e2e_unmatched += unmatched;

    printf("%s: %zu boxes, water IoU vs ref new %.5f old %.5f%s\n", path.filename().c_str(),
           small.boxes.size(), iou_new.back(), iou_old.back(),
           gt_new.empty()
               ? ""
               : cv::format(", vs labels new %.5f old %.5f", gt_new.back(), gt_old.back()).c_str());
  }

  printf("\n%zu frames, network %dx%d\n", frames.size(), net_w, net_h);
  if (undist_png_frames)
    printf("cv::remap here vs --undist pngs: %ld differing bytes over %ld frames\n",
           undist_png_mismatch, undist_png_frames);
  printf(
      "[tensor] map convertMaps(CV_16SC2) vs remap(CV_16SC2)+letterbox: %ld / %ld differ, "
      "max %d levels\n",
      converted_vs_remap.mismatches, converted_vs_remap.elements, converted_vs_remap.max_diff);
  printf(
      "[tensor] map CV_32FC2 vs remap(CV_32FC2)+letterbox:          %ld / %ld differ, "
      "max %d levels\n",
      float_vs_remapf.mismatches, float_vs_remapf.elements, float_vs_remapf.max_diff);
  printf(
      "[tensor] map CV_32FC2 vs remap(CV_16SC2)+letterbox:          %ld / %ld differ, "
      "max %d levels\n",
      float_vs_remap.mismatches, float_vs_remap.elements, float_vs_remap.max_diff);
  printf("[ref] class mask %dx%d: %ld / %ld cells differ; full resolution: %ld differ\n", kGridW,
         kGridH, ref_mismatch_small, ref_cells, ref_mismatch_full);
  printf("[ref] boxes: max coordinate diff %.5f px, %d unmatched; instance maps max diff %d\n",
         worst_box, ref_unmatched, worst_seg);
  printf("[ref] water IoU vs reference, median (min): new %.5f (%.5f), old %.5f (%.5f)\n",
         median(iou_new), *std::min_element(iou_new.begin(), iou_new.end()), median(iou_old),
         *std::min_element(iou_old.begin(), iou_old.end()));
  printf(
      "[e2e] forward(U) vs forward(raw)+map: %ld / %ld cells differ, max box diff %.5f px, %ld "
      "unmatched; pinned vs pageable: %ld cells differ\n",
      e2e_mismatch, e2e_cells, worst_e2e_box, e2e_unmatched, pinned_mismatch);
  if (!gt_new.empty())
    printf(
        "[gt] water IoU vs labels, median: new %.5f old %.5f; border-snapped: new %.5f old "
        "%.5f\n",
        median(gt_new), median(gt_old), median(gt_new_snap), median(gt_old_snap));
  return 0;
}
