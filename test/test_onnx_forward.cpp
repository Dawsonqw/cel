#include <gtest/gtest.h>
#include <string>
#include <glog/logging.h>
#include "Parser/onnx_parser.hpp"
#include "Parser/tensor.hpp"


TEST(OnnxParser, test_onnx_parser){
    google::InitGoogleLogging("test_onnx_forward");
    google::SetStderrLogging(google::INFO);
    const std::string file="/root/workspace/cel/test/resnet18.onnx";
    cel::Model model;
    cel::OnnxParser parser(file);
    parser.parse(&model);
    std::map<std::string,cel::tensor_vec<float>> inputs;
    cel::Tensor<float> input_tensor({3,224,224});
    input_tensor.RandU(-1.0f,1.0f);
    cel::tensor_vec<float> vec_inputs;
    vec_inputs.push_back(std::make_shared<cel::Tensor<float>>(input_tensor));
    inputs["input"]=vec_inputs;
    model.forward(inputs);
}