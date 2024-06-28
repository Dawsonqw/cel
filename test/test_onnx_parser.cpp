#include <gtest/gtest.h>
#include <string>
#include <glog/logging.h>
#include "Parser/onnx_parser.hpp"


TEST(OnnxParser, test_onnx_parser){
    // 初始化log
    google::InitGoogleLogging("test_onnx_parser");
    google::SetStderrLogging(google::INFO);
    const std::string file="/root/workspace/cel/test/resnet18.onnx";
    cel::Model model;
    cel::OnnxParser parser(file);
    parser.parse(&model);
}