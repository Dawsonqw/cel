#ifndef PARSER_COMPUTE_GEMM_H
#define PARSER_COMPUTE_GEMM_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"
#include "Parser/tensor_utils.hpp"

namespace cel{

    template<typename T>
    void gemm_compute(const tensor_vec<T>&input_a,const tensor_vec<T>&input_b,const tensor_vec<T>& input_c,tensor_vec<T>&output,float alpha,float beta,int64_t transB){
        LOG(INFO)<<"gemm compute start";
        int32_t batch_a=input_a.size();
        LOG_IF(FATAL,batch_a<1)<<"input size must be greater than 1";
        output.resize(batch_a);
        std::shared_ptr<Tensor<T>> input_tensor_b=input_b[0];
        std::shared_ptr<Tensor<T>> beta_tensor=std::make_shared<Tensor<T>>();
        if(beta!=0){
            beta_tensor=input_c[0];
        }
        for(int32_t index=0;index<batch_a;index++){
            std::shared_ptr<Tensor<T>> input_tensor_a=input_a[index];
            if(transB!=0){
                input_tensor_b->Transpose();
            }
            std::shared_ptr<Tensor<T>> output_tensor=cel::matmul(input_tensor_a,input_tensor_b);
            if(alpha!=1){
                std::shared_ptr<Tensor<T>> alpha_tensor=std::make_shared<Tensor<T>>(output_tensor->shapes());
                alpha_tensor->Fill(alpha);
                output_tensor=cel::mul(output_tensor,alpha_tensor);
            }
            if(beta!=0){
                output_tensor=cel::add(output_tensor,beta_tensor);
            }
            output[index]=output_tensor;
        }
        LOG(INFO)<<"gemm compute end";
    }
}
#endif