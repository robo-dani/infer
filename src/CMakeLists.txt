add_library(infer SHARED
    infer.cu
    yolo.cu
)


link_directories(/usr/lib/aarch64-linux-gnu/)


target_link_libraries(infer
    ${THIRD_PARTY_LIBS}
    nvinfer
    nvinfer_plugin
    nvonnxparser
)