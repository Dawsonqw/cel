#ifndef PARSER_COMPUTE_FLATTEN_H
#define PARSER_COMPUTE_FLATTEN_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"

namespace cel{

    template<typename T>
    void flatten_compute(const tensor_vec<T>&input,tensor_vec<T>&output,int64_t axis){
        LOG(INFO)<<"flatten compute done";
        for(auto tensor:input){
            tensor->Flatten(axis-1);
            output.push_back(tensor);
        }
        LOG(INFO)<<"flatten compute done";
    }
}
#endif