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

add_subdirectory(Compute)
add_library(onnx_parser STATIC ${_sources}) 