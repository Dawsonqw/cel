#ifndef PARSER_COMPUTE_ADD_H
#define PARSER_COMPUTE_ADD_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"
#include "Parser/tensor_utils.hpp"

namespace cel{

    template<typename T>
    void add_compute(const tensor_vec<T>&input_first,const tensor_vec<T>&input_second,tensor_vec<T>&output){
        LOG(INFO)<<"add compute start";
        int32_t first_batch=input_first.size();
        int32_t second_batch=input_second.size();
        LOG_IF(FATAL,first_batch!=second_batch)<<"input_first size must be equal to input_second size";
        output.resize(first_batch);
        for(int32_t index=0;index<first_batch;index++){
            std::shared_ptr<Tensor<T>> first_tensor=input_first[index];
            std::shared_ptr<Tensor<T>> second_tensor=input_second[index];
            output[index]=cel::add(first_tensor,second_tensor);
        }
        LOG(INFO)<<"add compute end";
    }
}
#endif