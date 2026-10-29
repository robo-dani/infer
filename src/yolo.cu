#include <cstring>

#include "check.hpp"
#include "infer.hpp"
#include "preprocess.hpp"
#include "segment.hpp"
#include "yolo.hpp"

namespace yolo {

using namespace std;

#define GPU_BLOCK_THREADS 512

const int NUM_BOX_ELEMENT = 8;  // left, top, right, bottom, confidence, class,
                                // keepflag, row_index(output)
const int MAX_IMAGE_BOXES = 1024;
static __global__ void decode_kernel_common(float *predict, int num_bboxes, int num_classes,
                                            int output_cdim, float confidence_threshold,
                                            Matrix23 invert_affine_matrix, float *parray,
                                            int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float objectness = pitem[4];
  if (objectness < confidence_threshold) return;

  float *class_confidence = pitem + 5;
  float confidence = *class_confidence++;
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }

  confidence *= objectness;
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width = *pitem++;
  float height = *pitem++;
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;  // 1 = keep, 0 = ignore
}

static __global__ void decode_kernel_v8(float *predict, int num_bboxes, int num_classes,
                                        int output_cdim, float confidence_threshold,
                                        Matrix23 invert_affine_matrix, float *parray,
                                        int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float *class_confidence = pitem + 4;
  float confidence = *class_confidence++;
  int label = 0;
  for (int i = 1; i < num_classes; ++i, ++class_confidence) {
    if (*class_confidence > confidence) {
      confidence = *class_confidence;
      label = i;
    }
  }
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float cx = *pitem++;
  float cy = *pitem++;
  float width = *pitem++;
  float height = *pitem++;
  float left = cx - width * 0.5f;
  float top = cy - height * 0.5f;
  float right = cx + width * 0.5f;
  float bottom = cy + height * 0.5f;
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = label;
  *pout_item++ = 1;  // 1 = keep, 0 = ignore
  *pout_item++ = position;
}

// YOLO26 end-to-end head: each row is already a final detection
// [x1, y1, x2, y2, confidence, class, mask coefficients...] -- the model
// de-duplicates internally, so rows only need thresholding, no NMS pass.
static __global__ void decode_kernel_v26seg(float *predict, int num_bboxes, int output_cdim,
                                            float confidence_threshold,
                                            Matrix23 invert_affine_matrix, float *parray,
                                            int MAX_IMAGE_BOXES) {
  int position = blockDim.x * blockIdx.x + threadIdx.x;
  if (position >= num_bboxes) return;

  float *pitem = predict + output_cdim * position;
  float confidence = pitem[4];
  if (confidence < confidence_threshold) return;

  int index = atomicAdd(parray, 1);
  if (index >= MAX_IMAGE_BOXES) return;

  float left = pitem[0];
  float top = pitem[1];
  float right = pitem[2];
  float bottom = pitem[3];
  affine_project(invert_affine_matrix, left, top, &left, &top);
  affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

  float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
  *pout_item++ = left;
  *pout_item++ = top;
  *pout_item++ = right;
  *pout_item++ = bottom;
  *pout_item++ = confidence;
  *pout_item++ = pitem[5];  // class label, provided directly by the head
  *pout_item++ = 1;         // 1 = keep, 0 = ignore
  *pout_item++ = position;  // row index (for mask coefficients)
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, float bleft,
                                float btop, float bright, float bbottom) {
  float cleft = max(aleft, bleft);
  float ctop = max(atop, btop);
  float cright = min(aright, bright);
  float cbottom = min(abottom, bbottom);

  float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
  if (c_area == 0.0f) return 0.0f;

  float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
  float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
  return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float *bboxes, int MAX_IMAGE_BOXES, float threshold) {
  int position = (blockDim.x * blockIdx.x + threadIdx.x);
  int count = min((int)*bboxes, MAX_IMAGE_BOXES);
  if (position >= count) return;

  // left, top, right, bottom, confidence, class, keepflag
  float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
  for (int i = 0; i < count; ++i) {
    float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
    if (i == position || pcurrent[5] != pitem[5]) continue;

    if (pitem[4] >= pcurrent[4]) {
      if (pitem[4] == pcurrent[4] && i < position) continue;

      float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1],
                          pitem[2], pitem[3]);

      if (iou > threshold) {
        pcurrent[6] = 0;  // 1=keep, 0=ignore
        return;
      }
    }
  }
}

static dim3 grid_dims(int numJobs) {
  int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
  return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs) {
  return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
}

static void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, int output_cdim,
                                  float confidence_threshold, float nms_threshold,
                                  const Matrix23 &invert_affine_matrix, float *parray,
                                  int MAX_IMAGE_BOXES, Type type, cudaStream_t stream) {
  auto grid = grid_dims(num_bboxes);
  auto block = block_dims(num_bboxes);

  if (type == Type::V26Seg) {
    checkKernel(decode_kernel_v26seg<<<grid, block, 0, stream>>>(
        predict, num_bboxes, output_cdim, confidence_threshold, invert_affine_matrix, parray,
        MAX_IMAGE_BOXES));
    return;  // end-to-end head: model already de-duplicated, skip NMS
  } else if (type == Type::V8) {
    checkKernel(decode_kernel_v8<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
        parray, MAX_IMAGE_BOXES));
  } else {
    checkKernel(decode_kernel_common<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, output_cdim, confidence_threshold, invert_affine_matrix,
        parray, MAX_IMAGE_BOXES));
  }

  grid = grid_dims(MAX_IMAGE_BOXES);
  block = block_dims(MAX_IMAGE_BOXES);
  checkKernel(fast_nms_kernel<<<grid, block, 0, stream>>>(parray, MAX_IMAGE_BOXES, nms_threshold));
}

const char *type_name(Type type) {
  switch (type) {
    case Type::V5:
      return "YoloV5";
    case Type::V3:
      return "YoloV3";
    case Type::V7:
      return "YoloV7";
    case Type::X:
      return "YoloX";
    case Type::V8:
      return "YoloV8";
    case Type::V26Seg:
      return "Yolo26Seg";
    default:
      return "Unknow";
  }
}

InstanceSegmentMap::InstanceSegmentMap(int width, int height) {
  this->width = width;
  this->height = height;
  this->data = new unsigned char[width * height];
}

InstanceSegmentMap::~InstanceSegmentMap() {
  delete[] this->data;
  this->data = nullptr;
  this->width = 0;
  this->height = 0;
}

class InferImpl : public Infer {
 public:
  shared_ptr<trt::Infer> trt_;
  string engine_file_;
  Type type_;
  float confidence_threshold_;
  float nms_threshold_;
  cudaStream_t own_stream_ = nullptr;  // used when the caller passes no stream
  vector<shared_ptr<trt::Memory<unsigned char>>> preprocess_buffers_;  // pageable frames, copied
  trt::Memory<float> input_buffer_, bbox_predict_, output_boxarray_;
  trt::Memory<float> segment_predict_;
  int memory_batch_ = 0;  // batch the buffers above are sized and bound for
  int network_input_width_, network_input_height_;
  Norm normalize_;
  vector<int> input_dims_;
  vector<int> bbox_head_dims_;
  vector<int> segment_head_dims_;
  int num_classes_ = 0;
  int mask_coef_offset_ = 0;  // offset of mask coefficients within a bbox row
  bool has_segment_ = false;
  bool isdynamic_model_ = false;

  trt::Memory<float> source_map_;
  int source_map_width_ = 0, source_map_height_ = 0;
  int class_mask_width_ = 0, class_mask_height_ = 0;

  // Segmentation scratch. These only grow between frames, when nothing on the stream uses them.
  vector<SegInstance> instances_;
  trt::Memory<SegInstance> instance_params_;
  trt::Memory<float> instance_logits_;
  trt::Memory<unsigned char> segment_output_;

  virtual ~InferImpl() {
    if (own_stream_) checkRuntime(cudaStreamDestroy(own_stream_));
  }

  void adjust_memory(int batch_size) {
    // the inference batch_size
    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    input_buffer_.gpu(batch_size * input_numel);
    bbox_predict_.gpu(batch_size * bbox_head_dims_[1] * bbox_head_dims_[2]);
    output_boxarray_.gpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
    output_boxarray_.cpu(batch_size * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT));
    if (has_segment_)
      segment_predict_.gpu(batch_size * segment_head_dims_[1] * segment_head_dims_[2] *
                           segment_head_dims_[3]);

    if ((int)preprocess_buffers_.size() < batch_size) {
      for (int i = preprocess_buffers_.size(); i < batch_size; ++i)
        preprocess_buffers_.push_back(make_shared<trt::Memory<unsigned char>>());
    }

    // tensor addresses only change here
    trt_->setTensorAddress("images", input_buffer_.gpu());
    trt_->setTensorAddress("output0", bbox_predict_.gpu());
    if (has_segment_) trt_->setTensorAddress("output1", segment_predict_.gpu());
    memory_batch_ = batch_size;
  }

  void preprocess(int ibatch, const Image &image, AffineMatrix &affine, cudaStream_t stream) {
    SourceMap source_map;
    if (source_map_width_ > 0)
      source_map = {(const float2 *)source_map_.gpu(), source_map_width_, source_map_height_};
    affine.compute(source_map.map ? source_map.width : image.width,
                   source_map.map ? source_map.height : image.height, network_input_width_,
                   network_input_height_);

    // A frame in page-locked, mapped or device memory is read in place, anything else is copied
    // over. Checking both ends keeps a frame that overruns its allocation off the in-place path.
    size_t size_image = (size_t)image.width * image.height * image.channels;
    const uint8_t *image_device = nullptr;
    cudaPointerAttributes first, last;
    if (cudaPointerGetAttributes(&first, image.bgrptr) == cudaSuccess &&
        cudaPointerGetAttributes(&last, (const uint8_t *)image.bgrptr + size_image - 1) ==
            cudaSuccess &&
        first.type != cudaMemoryTypeUnregistered && first.devicePointer != nullptr &&
        (const uint8_t *)last.devicePointer ==
            (const uint8_t *)first.devicePointer + size_image - 1) {
      image_device = (const uint8_t *)first.devicePointer;
    } else {
      cudaGetLastError();  // older runtimes report pageable memory as an error
      uint8_t *copy = preprocess_buffers_[ibatch]->gpu(size_image);
      checkRuntime(
          cudaMemcpyAsync(copy, image.bgrptr, size_image, cudaMemcpyHostToDevice, stream));
      image_device = copy;
    }

    size_t input_numel = network_input_width_ * network_input_height_ * 3;
    float *input_device = input_buffer_.gpu() + ibatch * input_numel;
    warp_affine_bilinear_and_normalize_plane(
        image_device, image.width * image.channels, image.channels, image.width, image.height,
        source_map, input_device, network_input_width_, network_input_height_, affine.d2i, 114,
        normalize_, stream);
  }

  bool load(const string &engine_file, Type type, float confidence_threshold, float nms_threshold) {
    trt_ = trt::load(engine_file);
    if (trt_ == nullptr) return false;

    trt_->print();

    this->type_ = type;
    this->confidence_threshold_ = confidence_threshold;
    this->nms_threshold_ = nms_threshold;

    // by name: TensorRT does not promise any order of the IO tensors
    has_segment_ = type == Type::V26Seg;
    vector<string> tensors = {"images", "output0"};
    if (has_segment_) tensors.push_back("output1");
    for (auto &name : tensors) {
      if (trt_->dtype(name) != trt::DType::FLOAT) {
        INFO("Tensor %s must be float32", name.c_str());
        return false;
      }
    }
    input_dims_ = trt_->static_dims("images");
    bbox_head_dims_ = trt_->static_dims("output0");
    if (has_segment_) {
      segment_head_dims_ = trt_->static_dims("output1");
      if (segment_head_dims_[1] > kMaxProtoChannels) {
        INFO("%d mask prototypes, at most %d are supported", segment_head_dims_[1],
             kMaxProtoChannels);
        return false;
      }
    }
    network_input_width_ = input_dims_[3];
    network_input_height_ = input_dims_[2];
    isdynamic_model_ = trt_->has_dynamic_dim();

    if (type == Type::V5 || type == Type::V3 || type == Type::V7) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      num_classes_ = bbox_head_dims_[2] - 5;
    } else if (type == Type::V8) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      num_classes_ = bbox_head_dims_[2] - 4;
    } else if (type == Type::V26Seg) {
      normalize_ = Norm::alpha_beta(1 / 255.0f, 0.0f, ChannelType::SwapRB);
      // end-to-end rows: [x1, y1, x2, y2, conf, cls, mask coefficients...];
      // class ids come straight from the row, num_classes_ is not involved
      // in decoding
      num_classes_ = 0;
      mask_coef_offset_ = 6;
    } else if (type == Type::X) {
      // float mean[] = {0.485, 0.456, 0.406};
      // float std[]  = {0.229, 0.224, 0.225};
      // normalize_ = Norm::mean_std(mean, std, 1/255.0f, ChannelType::SwapRB);
      normalize_ = Norm::None();
      num_classes_ = bbox_head_dims_[2] - 5;
    } else {
      INFO("Unsupport type %d", type);
      return false;
    }

    // a stream of our own: TensorRT on the legacy default stream adds device-wide syncs
    checkRuntime(cudaStreamCreateWithFlags(&own_stream_, cudaStreamNonBlocking));
    return true;
  }

  virtual bool set_class_mask_size(int width, int height) override {
    if (width < 0 || height < 0 || (width == 0) != (height == 0)) return false;
    if (width > 0 && !has_segment_) return false;
    class_mask_width_ = width;
    class_mask_height_ = height;
    return true;
  }

  virtual bool set_source_map(int width, int height, const float *map_xy) override {
    if (map_xy == nullptr) {
      source_map_.release_gpu();
      source_map_width_ = source_map_height_ = 0;
      return true;
    }
    if (width <= 0 || height <= 0) return false;

    // on our stream and waited for: a plain cudaMemcpy from pageable memory may return before
    // the data lands, and the legacy stream does not order against non-blocking ones
    size_t numel = (size_t)width * height * 2;
    float *map_device = source_map_.gpu(numel);
    checkRuntime(cudaMemcpyAsync(map_device, map_xy, numel * sizeof(float),
                                 cudaMemcpyHostToDevice, own_stream_));
    checkRuntime(cudaStreamSynchronize(own_stream_));
    source_map_width_ = width;
    source_map_height_ = height;
    return true;
  }

  virtual bool forward(const Image &image, Detections &out, void *stream = nullptr) override {
    return infer(&image, 1, &out, stream);
  }

  virtual bool forwards(const vector<Image> &images, vector<Detections> &out,
                        void *stream = nullptr) override {
    out.resize(images.size());
    if (images.empty()) return true;
    return infer(images.data(), images.size(), out.data(), stream);
  }

  bool infer(const Image *images, int num_image, Detections *outs, void *stream) {
    for (int i = 0; i < num_image; ++i) {
      if (images[i].bgrptr == nullptr || images[i].width <= 0 || images[i].height <= 0 ||
          (images[i].channels != 3 && images[i].channels != 4)) {
        INFO("Invalid image %d: %p %dx%d, %d channels", i, images[i].bgrptr, images[i].width,
             images[i].height, images[i].channels);
        return false;
      }
    }

    int infer_batch_size = input_dims_[0];
    if (infer_batch_size != num_image) {
      if (isdynamic_model_) {
        infer_batch_size = num_image;
        auto input_dims = input_dims_;
        input_dims[0] = num_image;
        if (!trt_->set_run_dims("images", input_dims)) return false;
      } else {
        if (infer_batch_size < num_image) {
          INFO(
              "When using static shape model, number of images[%d] must be "
              "less than or equal to the maximum batch[%d].",
              num_image, infer_batch_size);
          return false;
        }
      }
    }
    if (infer_batch_size != memory_batch_) adjust_memory(infer_batch_size);

    cudaStream_t stream_ = stream ? (cudaStream_t)stream : own_stream_;
    vector<AffineMatrix> affine_matrixs(num_image);
    for (int i = 0; i < num_image; ++i) preprocess(i, images[i], affine_matrixs[i], stream_);

    if (!trt_->forward(stream_)) {
      INFO("Failed to tensorRT forward.");
      return false;
    }

    float *bbox_output_device = bbox_predict_.gpu();
    for (int ib = 0; ib < num_image; ++ib) {
      float *boxarray_device =
          output_boxarray_.gpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
      float *image_based_bbox_output =
          bbox_output_device + ib * (bbox_head_dims_[1] * bbox_head_dims_[2]);
      checkRuntime(cudaMemsetAsync(boxarray_device, 0, sizeof(int), stream_));
      decode_kernel_invoker(image_based_bbox_output, bbox_head_dims_[1], num_classes_,
                            bbox_head_dims_[2], confidence_threshold_, nms_threshold_,
                            affine_matrixs[ib].d2i, boxarray_device, MAX_IMAGE_BOXES, type_,
                            stream_);
    }
    checkRuntime(cudaMemcpyAsync(output_boxarray_.cpu(), output_boxarray_.gpu(),
                                 output_boxarray_.gpu_bytes(), cudaMemcpyDeviceToHost, stream_));
    checkRuntime(cudaStreamSynchronize(stream_));

    instances_.clear();
    for (int ib = 0; ib < num_image; ++ib) {
      float *parray = output_boxarray_.cpu() + ib * (32 + MAX_IMAGE_BOXES * NUM_BOX_ELEMENT);
      int count = min(MAX_IMAGE_BOXES, (int)*parray);
      BoxArray &output = outs[ib].boxes;
      output.clear();
      output.reserve(count);
      for (int i = 0; i < count; ++i) {
        float *pbox = parray + 1 + i * NUM_BOX_ELEMENT;
        int label = pbox[5];
        int keepflag = pbox[6];
        if (keepflag != 1) continue;

        output.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
        if (!has_segment_) continue;

        int row_index = pbox[7];
        const Matrix23 &i2d = affine_matrixs[ib].i2d;
        float scale_to_predict_x = segment_head_dims_[3] / (float)network_input_width_;
        float scale_to_predict_y = segment_head_dims_[2] / (float)network_input_height_;
        SegInstance instance;
        instance.coefficients = bbox_output_device +
                                (ib * bbox_head_dims_[1] + row_index) * bbox_head_dims_[2] +
                                mask_coef_offset_;
        instance.proto = segment_predict_.gpu() +
                         ib * segment_head_dims_[1] * segment_head_dims_[2] * segment_head_dims_[3];
        instance.left = pbox[0];
        instance.top = pbox[1];
        instance.right = pbox[2];
        instance.bottom = pbox[3];
        instance.proto_scale_x = i2d.v[0] * scale_to_predict_x;
        instance.proto_offset_x = i2d.v[2] * scale_to_predict_x - 0.5f;
        instance.proto_scale_y = i2d.v[4] * scale_to_predict_y;
        instance.proto_offset_y = i2d.v[5] * scale_to_predict_y - 0.5f;
        instance.class_label = label;
        // the box at prototype resolution
        instance.mask_width = (pbox[2] - pbox[0]) * instance.proto_scale_x + 0.5f;
        instance.mask_height = (pbox[3] - pbox[1]) * instance.proto_scale_y + 0.5f;
        if (instance.mask_width <= 0 || instance.mask_height <= 0)
          instance.mask_width = instance.mask_height = 0;
        instances_.push_back(instance);
      }
    }

    if (has_segment_) render_segments(images, num_image, outs, stream_);
    for (int ib = 0; ib < num_image && class_mask_width_ == 0; ++ib) outs[ib].class_mask = {};
    return true;
  }

  // Instance maps and class masks of every image in one pass: one upload of the instances, one
  // download of all masks
  void render_segments(const Image *images, int num_image, Detections *outs, cudaStream_t stream) {
    size_t class_mask_bytes = (size_t)class_mask_width_ * class_mask_height_;
    size_t total_bytes = class_mask_bytes * num_image;
    int max_mask_width = 0, max_mask_height = 0;
    for (auto &instance : instances_) {
      instance.mask_offset = total_bytes;
      total_bytes += (size_t)instance.mask_width * instance.mask_height;
      max_mask_width = max(max_mask_width, instance.mask_width);
      max_mask_height = max(max_mask_height, instance.mask_height);
    }
    if (total_bytes == 0) return;

    int count = instances_.size();
    ProtoShape proto{segment_head_dims_[1], segment_head_dims_[3], segment_head_dims_[2]};
    size_t plane = (size_t)proto.width * proto.height;
    uint8_t *output_device = segment_output_.gpu(total_bytes);
    uint8_t *output_host = segment_output_.cpu(total_bytes);
    SegInstance *instances_device = nullptr;
    float *logits = nullptr;
    if (count > 0) {
      memcpy(instance_params_.cpu(count), instances_.data(), count * sizeof(SegInstance));
      instances_device = instance_params_.gpu(count);
      checkRuntime(cudaMemcpyAsync(instances_device, instance_params_.cpu(),
                                   count * sizeof(SegInstance), cudaMemcpyHostToDevice, stream));
      logits = instance_logits_.gpu(count * plane);
      compute_instance_logits(instances_device, count, proto, logits, stream);
      if (max_mask_width > 0)
        render_instance_masks(instances_device, count, max_mask_width, max_mask_height, logits,
                              proto, output_device, stream);
    }
    if (class_mask_bytes > 0) {
      // instances_ holds the boxes of all images in order
      int first = 0;
      for (int ib = 0; ib < num_image; ++ib) {
        int image_count = outs[ib].boxes.size();
        int image_width = source_map_width_ > 0 ? source_map_width_ : images[ib].width;
        int image_height = source_map_width_ > 0 ? source_map_height_ : images[ib].height;
        render_class_mask(instances_device + first, image_count, logits + first * plane, proto,
                          image_width, image_height, output_device + ib * class_mask_bytes,
                          class_mask_width_, class_mask_height_, stream);
        first += image_count;
      }
    }
    checkRuntime(cudaMemcpyAsync(output_host, output_device, total_bytes, cudaMemcpyDeviceToHost,
                                 stream));
    checkRuntime(cudaStreamSynchronize(stream));

    int k = 0;
    for (int ib = 0; ib < num_image; ++ib) {
      if (class_mask_bytes > 0) {
        ClassMask &mask = outs[ib].class_mask;
        mask.width = class_mask_width_;
        mask.height = class_mask_height_;
        mask.data.assign(output_host + ib * class_mask_bytes,
                         output_host + (ib + 1) * class_mask_bytes);
      }
      for (auto &box : outs[ib].boxes) {
        const SegInstance &instance = instances_[k++];
        if (instance.mask_width == 0) continue;
        box.seg = make_shared<InstanceSegmentMap>(instance.mask_width, instance.mask_height);
        memcpy(box.seg->data, output_host + instance.mask_offset,
               instance.mask_width * instance.mask_height);
      }
    }
  }
};

Infer *loadraw(const std::string &engine_file, Type type, float confidence_threshold,
               float nms_threshold) {
  InferImpl *impl = new InferImpl();
  if (!impl->load(engine_file, type, confidence_threshold, nms_threshold)) {
    delete impl;
    impl = nullptr;
  }
  return impl;
}

shared_ptr<Infer> load(const string &engine_file, Type type, float confidence_threshold,
                       float nms_threshold) {
  return std::shared_ptr<InferImpl>(
      (InferImpl *)loadraw(engine_file, type, confidence_threshold, nms_threshold));
}

};  // namespace yolo