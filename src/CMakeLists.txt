add_library(infer SHARED
    infer.cu
    yolo.cu
)




target_link_libraries(infer
    ${THIRD_PARTY_LIBS}
    nvinfer
    nvinfer_plugin
    nvparsers
    nvonnxparser
)