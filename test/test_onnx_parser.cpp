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
    std::map<std::string,cel::Tensor<float>> inputs;
    cel::Tensor<float> input_tensor({3,224,224});
    input_tensor.RandU(-1.0f,1.0f);
    inputs["input"]=input_tensor;
    model.forward(inputs);
}