# Checks and benchmarks against real frames; they call internal kernels, so they see src/
find_package(CUDAToolkit REQUIRED)

foreach(target check_v26seg bench_v26seg)
    add_executable(${target} ${target}.cpp)
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${target} infer ${THIRD_PARTY_LIBS} CUDA::cudart)
endforeach()
