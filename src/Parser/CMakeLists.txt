file(GLOB _sources
          edge.cc
          node.cc
          model.cc
          attribute.cc
          tensor.cc          
          onnx.pb.cc
          onnx_parser.cc
    )

add_library(onnx_parser SHARED  ${PROTOBUF_LIBRARY_DEBUG} ${_sources})