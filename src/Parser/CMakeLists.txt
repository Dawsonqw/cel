file(GLOB _sources
          edge.cc
          node.cc
          model.cc
          attribute.cc
          tensor.cc          
          tensor_utils.cc          
          onnx.pb.cc
          onnx_parser.cc
          OnnxNodes/*.cc
    )

set(
    LIBS
    glog::glog
)

add_library(onnx_parser STATIC ${_sources})
add_subdirectory(Compute)