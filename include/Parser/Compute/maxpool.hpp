#ifndef PARSER_COMPUTE_MAXPOOL_H
#define PARSER_COMPUTE_MAXPOOL_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"

namespace cel{

    template<typename T>
    void maxpool_compute(const tensor_vec<T>&input,tensor_vec<T>&output,int64_t ceil_mode,std::vector<int64_t> kernel_shape,std::vector<int64_t> pads,
                                            std::vector<int64_t> strides,std::vector<int64_t> dilations){
            LOG(INFO)<<"maxpool compute done";
    }
}
#endif