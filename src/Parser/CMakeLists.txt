file(GLOB _sources
        #   onnx.pb.cc
        #   onnx_parser.cc
          edge.cc
          node.cc
          model.cc
          attribute.cc
    )

add_library(onnx_parser STATIC ${_sources})