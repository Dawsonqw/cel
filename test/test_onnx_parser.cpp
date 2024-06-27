#include <gtest/gtest.h>
#include <string>
#include <glog/logging.h>
#include "Parser/onnx_parser.hpp"


TEST(OnnxParser, test_onnx_parser){
    const std::string file="/root/workspace/cel/test/resnet18.onnx";
    cel::OnnxParser parser(file);
    parser.parse();
}