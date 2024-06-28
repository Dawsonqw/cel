file(GLOB _sources
          edge.cc
          node.cc
          model.cc
          attribute.cc
          tensor.cc          
          onnx.pb.cc
          onnx_parser.cc
    )

set(
    LIBS
    glog::glog
)

add_library(onnx_parser STATIC ${_sources})