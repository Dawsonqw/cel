#ifndef PARSER_COMPUTE_RELU_H
#define PARSER_COMPUTE_RELU_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"

namespace cel{

    template<typename T>
    void relu_compute(const tensor_vec<T>&input,tensor_vec<T>&output){
        LOG(INFO)<<"relu compute done";
    }
}
#endif