cmake_minimum_required(VERSION 3.28)
project(infer LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(OpenCV REQUIRED)


#### source and test ############
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include/")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIR}

)


set(TENSORRT_LIBRARY_DIR "/usr/local/cuda/lib64/")
set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    # ${TENSORRT_LIBRARY_DIR}
)


set(SOURCES_CODE
    src/main.cpp
    src/yolo.cu
    src/infer.cu
)


add_executable(single ${SOURCES_CODE})

target_link_libraries(single ${THIRD_PARTY_LIBS}
)

link_directories(/usr/lib/aarch64-linux-gnu/)
target_link_libraries(single nvinfer nvinfer_plugin nvparsers nvonnxparser)
