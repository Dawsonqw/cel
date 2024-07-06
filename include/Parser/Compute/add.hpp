#ifndef PARSER_COMPUTE_ADD_H
#define PARSER_COMPUTE_ADD_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"

namespace cel{

    template<typename T>
    void add_compute(const tensor_vec<T>&input_first,const tensor_vec<T>&input_second,tensor_vec<T>&output){
        LOG(INFO)<<"add compute done";
    }
}
#endif