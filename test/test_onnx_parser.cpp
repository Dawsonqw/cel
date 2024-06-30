#include <gtest/gtest.h>
#include <string>
#include <glog/logging.h>
#include "Parser/onnx_parser.hpp"
#include "Parser/tensor.hpp"


TEST(OnnxParser, test_onnx_parser){
    google::InitGoogleLogging("test_onnx_parser");
    google::SetStderrLogging(google::INFO);
    const std::string file="/root/workspace/cel/test/resnet18.onnx";
    cel::Model model;
    cel::OnnxParser parser(file);
    parser.parse(&model);
}