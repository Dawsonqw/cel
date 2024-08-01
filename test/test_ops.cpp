#include <gtest/gtest.h>
#include "Parser/tensor.hpp"
#include "Parser/utils.hpp"
#include <glog/logging.h>
#include <vector>

#include "Parser/Compute/conv.hpp"

TEST(OpsTest, test_conv){
    google::InitGoogleLogging("test_conv");
    google::SetStderrLogging(google::INFO);

    cel::tensor_vec<float> inputs;
    inputs.push_back(std::make_shared<cel::Tensor<float>>());
    std::vector<int32_t> input_shapes({1,64,56,56});
    std::vector<int32_t> kernel_shapes({64,64,3,3});
    std::vector<int32_t> bias_shapes({64});
    cel::tensor_vec<float> kernels;
    for(int i=0;i<64;i++){
        kernels.push_back(std::make_shared<cel::Tensor<float>>());
    }
    cel::tensor_vec<float> bias;
    bias.push_back(std::make_shared<cel::Tensor<float>>());
    for(int index=0;index<inputs.size();index++){
        std::shared_ptr<cel::Tensor<float>> input = inputs[index];
        std::vector<int32_t> input_shape = std::vector<int32_t>(input_shapes.begin()+1,input_shapes.end());
        size_t offset=index*std::accumulate(input_shape.begin(),input_shape.end(),1,std::multiplies<int32_t>());
        input->load("/root/workspace/cel/build/conv_input.bin",input_shape,offset,false);
    }
    for(int index=0;index<kernels.size();index++){
        std::shared_ptr<cel::Tensor<float>> kernel = kernels[index];
        std::vector<int32_t> kernel_shape = std::vector<int32_t>(kernel_shapes.begin()+1,kernel_shapes.end());
        size_t offset=index*std::accumulate(kernel_shape.begin(),kernel_shape.end(),1,std::multiplies<int32_t>());
        kernel->load("/root/workspace/cel/build/conv_weight.bin",kernel_shape,offset,false);
    }
    for(int index=0;index<bias.size();index++){
        std::shared_ptr<cel::Tensor<float>> b = bias[index];
        std::vector<int32_t> b_shape = std::vector<int32_t>(bias_shapes.begin(),bias_shapes.end());
        size_t offset=index*std::accumulate(b_shape.begin(),b_shape.end(),1,std::multiplies<int32_t>());
        b->load("/root/workspace/cel/build/conv_bias.bin",b_shape,offset,false);
    }
    cel::tensor_vec<float> outputs;


    std::vector<int64_t> pads({1,1,1,1});
    std::vector<int64_t> stride({1,1});
    std::vector<int64_t> dilation({1,1});
    std::vector<int64_t> kernel_shape({3,3});

    cel::conv_compute<float>(inputs,kernels,bias,outputs,pads,stride,dilation,kernel_shape);

    for(int index=0;index<outputs.size();index++){
        for(int c=0;c<outputs[index]->channels();c++){
            for(int h=0;h<outputs[index]->rows();h++){
                for(int w=0;w<outputs[index]->cols();w++){
                    float value=outputs[index]->at(c,h,w);
                    LOG(INFO)<<"output["<<index<<"]["<<c<<"]["<<h<<"]["<<w<<"]="<<value;
                }
            }
        }
    }

}