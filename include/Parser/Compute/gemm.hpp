#ifndef PARSER_COMPUTE_GEMM_H
#define PARSER_COMPUTE_GEMM_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"

namespace cel{

    template<typename T>
    void gemm_compute(const tensor_vec<T>&input_a,const tensor_vec<T>&input_b,const tensor_vec<T>& input_c,tensor_vec<T>&output,float alpha,float beta,int64_t transB){
        LOG(INFO)<<"gemm compute done";
    }
}
#endif