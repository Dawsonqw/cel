#ifndef PARSER_COMPUTE_RELU_H
#define PARSER_COMPUTE_RELU_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"

namespace cel{

    template<typename T>
    void relu_compute(const tensor_vec<T>&input,tensor_vec<T>&output){
        LOG(INFO)<<"relu compute done";

        int32_t batch=input.size();
        output.resize(batch);
        for(int32_t index=0;index<batch;index++){
            std::shared_ptr<Tensor<T>> input_tensor=input[index];
            std::shared_ptr<Tensor<T>> output_tensor=std::make_shared<Tensor<T>>();
            output_tensor->set_size(input_tensor->raw_shapes());
            for(int32_t i=0;i<input_tensor->size();i++){
                output_tensor->index(i)=std::max(input_tensor->index(i),static_cast<T>(0));
            }
            output[index]=output_tensor;
        }
        LOG(INFO)<<"relu compute done";
    }
}
#endif