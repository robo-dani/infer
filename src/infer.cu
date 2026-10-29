
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <stdarg.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "check.hpp"
#include "infer.hpp"

namespace trt {

using namespace std;
using namespace nvinfer1;

#define Assert(op)                 \
  do {                             \
    bool cond = !(!(op));          \
    if (!cond) {                   \
      INFO("Assert failed, " #op); \
      abort();                     \
    }                              \
  } while (0)

#define Assertf(op, ...)                             \
  do {                                               \
    bool cond = !(!(op));                            \
    if (!cond) {                                     \
      INFO("Assert failed, " #op " : " __VA_ARGS__); \
      abort();                                       \
    }                                                \
  } while (0)

static string file_name(const string &path, bool include_suffix) {
  if (path.empty()) return "";

  int p = path.rfind('/');
  int e = path.rfind('\\');
  p = std::max(p, e);
  p += 1;

  // include suffix
  if (include_suffix) return path.substr(p);

  int u = path.rfind('.');
  if (u == -1) return path.substr(p);

  if (u <= p) u = path.size();
  return path.substr(p, u - p);
}

void __log_func(const char *file, int line, const char *fmt, ...) {
  va_list vl;
  va_start(vl, fmt);
  char buffer[2048];
  string filename = file_name(file, true);
  int n = snprintf(buffer, sizeof(buffer), "[%s:%d]: ", filename.c_str(), line);
  vsnprintf(buffer + n, sizeof(buffer) - n, fmt, vl);
  fprintf(stdout, "%s\n", buffer);
}

static std::string format_shape(const Dims &shape) {
  stringstream output;
  char buf[64];
  const char *fmts[] = {"%d", "x%d"};
  for (int i = 0; i < shape.nbDims; ++i) {
    snprintf(buf, sizeof(buf), fmts[i != 0], shape.d[i]);
    output << buf;
  }
  return output.str();
}

Timer::Timer() {
  checkRuntime(cudaEventCreate((cudaEvent_t *)&start_));
  checkRuntime(cudaEventCreate((cudaEvent_t *)&stop_));
}

Timer::~Timer() {
  checkRuntime(cudaEventDestroy((cudaEvent_t)start_));
  checkRuntime(cudaEventDestroy((cudaEvent_t)stop_));
}

void Timer::start(void *stream) {
  stream_ = stream;
  checkRuntime(cudaEventRecord((cudaEvent_t)start_, (cudaStream_t)stream_));
}

float Timer::stop(const char *prefix, bool print) {
  checkRuntime(cudaEventRecord((cudaEvent_t)stop_, (cudaStream_t)stream_));
  checkRuntime(cudaEventSynchronize((cudaEvent_t)stop_));

  float latency = 0;
  checkRuntime(cudaEventElapsedTime(&latency, (cudaEvent_t)start_, (cudaEvent_t)stop_));

  if (print) {
    printf("[%s]: %.5f ms\n", prefix, latency);
  }
  return latency;
}

BaseMemory::BaseMemory(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes) {
  reference(cpu, cpu_bytes, gpu, gpu_bytes);
}

void BaseMemory::reference(void *cpu, size_t cpu_bytes, void *gpu, size_t gpu_bytes) {
  release();

  if (cpu == nullptr || cpu_bytes == 0) {
    cpu = nullptr;
    cpu_bytes = 0;
  }

  if (gpu == nullptr || gpu_bytes == 0) {
    gpu = nullptr;
    gpu_bytes = 0;
  }

  this->cpu_ = cpu;
  this->cpu_capacity_ = cpu_bytes;
  this->cpu_bytes_ = cpu_bytes;
  this->gpu_ = gpu;
  this->gpu_capacity_ = gpu_bytes;
  this->gpu_bytes_ = gpu_bytes;

  this->owner_cpu_ = !(cpu && cpu_bytes > 0);
  this->owner_gpu_ = !(gpu && gpu_bytes > 0);
}

BaseMemory::~BaseMemory() { release(); }

void *BaseMemory::gpu_realloc(size_t bytes) {
  if (gpu_capacity_ < bytes) {
    release_gpu();

    gpu_capacity_ = bytes;
    checkRuntime(cudaMalloc(&gpu_, bytes));
    // checkRuntime(cudaMemset(gpu_, 0, size));
  }
  gpu_bytes_ = bytes;
  return gpu_;
}

void *BaseMemory::cpu_realloc(size_t bytes) {
  if (cpu_capacity_ < bytes) {
    release_cpu();

    cpu_capacity_ = bytes;
    checkRuntime(cudaMallocHost(&cpu_, bytes));
    Assert(cpu_ != nullptr);
    // memset(cpu_, 0, size);
  }
  cpu_bytes_ = bytes;
  return cpu_;
}

void BaseMemory::release_cpu() {
  if (cpu_) {
    if (owner_cpu_) {
      checkRuntime(cudaFreeHost(cpu_));
    }
    cpu_ = nullptr;
  }
  cpu_capacity_ = 0;
  cpu_bytes_ = 0;
}

void BaseMemory::release_gpu() {
  if (gpu_) {
    if (owner_gpu_) {
      checkRuntime(cudaFree(gpu_));
    }
    gpu_ = nullptr;
  }
  gpu_capacity_ = 0;
  gpu_bytes_ = 0;
}

void BaseMemory::release() {
  release_cpu();
  release_gpu();
}

class __native_nvinfer_logger : public ILogger {
 public:
  virtual void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kINTERNAL_ERROR) {
      INFO("NVInfer INTERNAL_ERROR: %s", msg);
      abort();
    } else if (severity == Severity::kERROR) {
      INFO("NVInfer: %s", msg);
    } else if (severity == Severity::kWARNING) {
      INFO("NVInfer WARNING: %s", msg);
    }
  }
};
static __native_nvinfer_logger gLogger;

template <typename _T>
static void destroy_nvidia_pointer(_T *ptr) {
  if (ptr) delete ptr;
}

static std::vector<uint8_t> load_file(const string &file) {
  ifstream in(file, ios::in | ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, ios::beg);
    data.resize(length);

    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

class __native_engine_context {
 public:
  virtual ~__native_engine_context() { destroy(); }

  bool construct(const void *pdata, size_t size) {
    destroy();

    if (pdata == nullptr || size == 0) return false;

    runtime_ = shared_ptr<IRuntime>(createInferRuntime(gLogger), destroy_nvidia_pointer<IRuntime>);
    if (runtime_ == nullptr) return false;

    engine_ = shared_ptr<ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size),
                                      destroy_nvidia_pointer<ICudaEngine>);
    if (engine_ == nullptr) return false;

    context_ = shared_ptr<IExecutionContext>(engine_->createExecutionContext(),
                                             destroy_nvidia_pointer<IExecutionContext>);
    return context_ != nullptr;
  }

 private:
  void destroy() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }

 public:
  shared_ptr<IExecutionContext> context_;
  shared_ptr<ICudaEngine> engine_;
  shared_ptr<IRuntime> runtime_ = nullptr;
};

class InferImpl : public Infer {
 public:
  shared_ptr<__native_engine_context> context_;
  unordered_map<string, int> binding_name_to_index_;
  unordered_map<string, void *> tensor_address_;

  // forward() replays enqueueV3 as a CUDA graph, which cuts the per-frame CPU launch cost from
  // milliseconds to microseconds. The graph bakes in tensor addresses and shapes, so any change
  // to them drops it and the next calls re-capture.
  cudaGraphExec_t graph_exec_ = nullptr;
  bool graph_warmed_up_ = false;
  bool graph_unsupported_ = false;

  virtual ~InferImpl() { invalidate_graph(); }

  void invalidate_graph() {
    if (graph_exec_) checkRuntime(cudaGraphExecDestroy(graph_exec_));
    graph_exec_ = nullptr;
    graph_warmed_up_ = false;
  }

  bool construct(const void *data, size_t size) {
    context_ = make_shared<__native_engine_context>();
    if (!context_->construct(data, size)) {
      return false;
    }

    setup();
    return true;
  }

  bool load(const string &file) {
    auto data = load_file(file);
    if (data.empty()) {
      INFO("An empty file has been loaded. Please confirm your file path: %s", file.c_str());
      return false;
    }
    return this->construct(data.data(), data.size());
  }

  void setup() {
    auto engine = this->context_->engine_;
    int nbBindings = engine->getNbIOTensors();

    binding_name_to_index_.clear();
    for (int i = 0; i < nbBindings; ++i) {
      const char *bindingName = engine->getIOTensorName(i);
      binding_name_to_index_[bindingName] = i;
    }
  }

  virtual int index(const std::string &name) override {
    auto iter = binding_name_to_index_.find(name);
    Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s",
            name.c_str());
    return iter->second;
  }

  virtual bool setTensorAddress(char const *name, void *data) override {
    auto iter = tensor_address_.find(name);
    if (iter != tensor_address_.end() && iter->second == data) return true;
    if (!this->context_->context_->setTensorAddress(name, data)) return false;
    tensor_address_[name] = data;
    invalidate_graph();
    return true;
  }

  virtual bool forward(void *stream) override {
    auto context = this->context_->context_;
    cudaStream_t s = (cudaStream_t)stream;
    // the legacy default stream cannot be captured
    if (s == nullptr || graph_unsupported_) return context->enqueueV3(s);
    if (graph_exec_) return cudaGraphLaunch(graph_exec_, s) == cudaSuccess;
    if (!graph_warmed_up_) {
      // TensorRT finishes deferred setup in the first enqueue after a change; keep it out of the
      // capture
      graph_warmed_up_ = true;
      return context->enqueueV3(s);
    }

    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    bool enqueued = false;
    if (err == cudaSuccess) {
      enqueued = context->enqueueV3(s);
      err = cudaStreamEndCapture(s, &graph);
    }
    if (enqueued && err == cudaSuccess) err = cudaGraphInstantiate(&graph_exec_, graph, 0);
    if (graph) checkRuntime(cudaGraphDestroy(graph));
    if (!enqueued || err != cudaSuccess) {
      // e.g. an engine with layers that synchronize with the host; plain enqueue still works
      INFO("NVInfer WARNING: CUDA graph capture failed (%s), using plain enqueueV3",
           cudaGetErrorString(err));
      cudaGetLastError();
      graph_exec_ = nullptr;
      graph_unsupported_ = true;
      return context->enqueueV3(s);
    }
    return cudaGraphLaunch(graph_exec_, s) == cudaSuccess;
  }

  virtual std::vector<int> run_dims(const std::string &name) override {
    return run_dims(index(name));
  }

  virtual std::vector<int> run_dims(int ibinding) override {
    auto name = this->context_->engine_->getIOTensorName(ibinding);
    auto dim = this->context_->context_->getTensorShape(name);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual std::vector<int> static_dims(const std::string &name) override {
    return static_dims(index(name));
  }

  virtual std::vector<int> static_dims(int ibinding) override {
    auto name = this->context_->engine_->getIOTensorName(ibinding);
    auto dim = this->context_->engine_->getTensorShape(name);
    return std::vector<int>(dim.d, dim.d + dim.nbDims);
  }

  virtual int num_bindings() override { return this->context_->engine_->getNbIOTensors(); }

  // virtual bool is_input(int ibinding) override {
  //   auto mode = engine->getTensorIOMode()
  //   return this->context_->engine_->bindingIsInput(ibinding);
  // }

  virtual bool set_run_dims(const std::string &name, const std::vector<int> &dims) override {
    return this->set_run_dims(index(name), dims);
  }

  virtual bool set_run_dims(int ibinding, const std::vector<int> &dims) override {
    if (dims.size() > (size_t)Dims::MAX_DIMS) return false;
    Dims d;
    d.nbDims = dims.size();
    for (int i = 0; i < d.nbDims; ++i) d.d[i] = dims[i];

    auto name = this->context_->engine_->getIOTensorName(ibinding);
    Dims current = this->context_->context_->getTensorShape(name);
    if (current.nbDims == d.nbDims && std::equal(d.d, d.d + d.nbDims, current.d)) return true;
    if (!this->context_->context_->setInputShape(name, d)) return false;
    invalidate_graph();
    return true;
  }

  virtual int numel(const std::string &name) override { return numel(index(name)); }

  virtual int numel(int ibinding) override {
    auto name = this->context_->engine_->getIOTensorName(ibinding);
    auto dim = this->context_->context_->getTensorShape(name);
    return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<int>());
  }

  virtual DType dtype(const std::string &name) override { return (DType)this->context_->engine_->getTensorDataType(name.c_str()); }

  // virtual DType dtype(int ibinding) override {
  //   return (DType)this->context_->engine_->getBindingDataType(ibinding);
  // }

  virtual bool has_dynamic_dim() override {
    // check if any input or output bindings have dynamic shapes
    // code from ChatGPT
    int numBindings = this->context_->engine_->getNbIOTensors();
    for (int i = 0; i < numBindings; ++i) {
      auto name = this->context_->engine_->getIOTensorName(i);
      nvinfer1::Dims dims = this->context_->engine_->getTensorShape(name);
      for (int j = 0; j < dims.nbDims; ++j) {
        if (dims.d[j] == -1) return true;
      }
    }
    return false;
  }

  virtual void print() override {
    INFO("Infer %p [%s]", this, has_dynamic_dim() ? "DynamicShape" : "StaticShape");

    // TensorRT does not order IO tensors by mode, so group them here
    auto engine = this->context_->engine_;
    vector<const char *> inputs, outputs;
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
      auto name = engine->getIOTensorName(i);
      if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        inputs.push_back(name);
      else
        outputs.push_back(name);
    }

    INFO("Inputs: %d", (int)inputs.size());
    for (int i = 0; i < (int)inputs.size(); ++i)
      INFO("\t%d.%s : shape {%s}", i, inputs[i],
           format_shape(engine->getTensorShape(inputs[i])).c_str());

    INFO("Outputs: %d", (int)outputs.size());
    for (int i = 0; i < (int)outputs.size(); ++i)
      INFO("\t%d.%s : shape {%s}", i, outputs[i],
           format_shape(engine->getTensorShape(outputs[i])).c_str());
  }
};

Infer *loadraw(const std::string &file) {
  InferImpl *impl = new InferImpl();
  if (!impl->load(file)) {
    delete impl;
    impl = nullptr;
  }
  return impl;
}

std::shared_ptr<Infer> load(const std::string &file) {
  return std::shared_ptr<InferImpl>((InferImpl *)loadraw(file));
}

std::string format_shape(const std::vector<int> &shape) {
  stringstream output;
  char buf[64];
  const char *fmts[] = {"%d", "x%d"};
  for (int i = 0; i < (int)shape.size(); ++i) {
    snprintf(buf, sizeof(buf), fmts[i != 0], shape[i]);
    output << buf;
  }
  return output.str();
}
};  // namespace trt
