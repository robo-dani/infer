cmake_minimum_required(VERSION 3.28)
project(infer LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)


set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


#### source and test ############

find_package(OpenCV REQUIRED)
set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)
link_directories(/usr/lib/aarch64-linux-gnu/)


add_subdirectory(src)
add_subdirectory(app)