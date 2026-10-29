#ifndef __CHECK_HPP__
#define __CHECK_HPP__

#include <cuda_runtime.h>

#include <cstdlib>

#include "infer.hpp"

#define checkRuntime(call)                                                                 \
  do {                                                                                     \
    auto ___call__ret_code__ = (call);                                                     \
    if (___call__ret_code__ != cudaSuccess) {                                              \
      INFO("CUDA Runtime error💥 %s # %s, code = %s [ %d ]", #call,                        \
           cudaGetErrorString(___call__ret_code__), cudaGetErrorName(___call__ret_code__), \
           ___call__ret_code__);                                                           \
      abort();                                                                             \
    }                                                                                      \
  } while (0)

#define checkKernel(...)                 \
  do {                                   \
    {                                    \
      (__VA_ARGS__);                     \
    }                                    \
    checkRuntime(cudaPeekAtLastError()); \
  } while (0)

#endif  // __CHECK_HPP__
