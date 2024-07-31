#ifndef PARSER_COMPUTE_GLOBALAVERAGEPOOL_H
#define PARSER_COMPUTE_GLOBALAVERAGEPOOL_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"

namespace cel{

    template<typename T>
    void globalaveragepool_compute(const tensor_vec<T>&input,tensor_vec<T>&output){
        LOG(INFO)<<"globalaveragepool compute done";
        // const arma::Cube<T>& inputData = input.data();
        // const arma::uword channels = inputData.n_slices;
        // const arma::uword rows = inputData.n_rows;
        // const arma::uword cols = inputData.n_cols;

        // // 初始化输出张量
        // arma::vec outputData(channels);
        // // 计算全局平均值
        // for (arma::uword c = 0; c < channels; ++c) {
        //     arma::mat slice = inputData.slice(c);
        //     T averageValue = accu(slice) / (rows * cols); // 计算平均值
        //     outputData(c) = averageValue;
        // }
        int32_t batch_size=input.size();
        LOG_IF(FATAL,batch_size<1)<<"input size must be greater than 1";
        int32_t channels=input[0]->raw_shapes()[0];
        int32_t height=input[0]->raw_shapes()[1];
        int32_t width=input[0]->raw_shapes()[2];
        output.resize(batch_size);
        for(int32_t index=0;index<batch_size;index++){
            std::shared_ptr<Tensor<T>> input_tensor=input[index];
            std::shared_ptr<Tensor<T>> output_tensor=std::make_shared<Tensor<T>>();
            output_tensor->set_size({channels,1,1});
            for(int32_t c=0;c<channels;c++){
                T sum=0;
                for(int32_t h=0;h<height;h++){
                    for(int32_t w=0;w<width;w++){
                        sum+=input_tensor->at(c,h,w);
                    }
                }
                output_tensor->set_data(c,0,0,sum/(height*width));
            }
            output[index]=output_tensor;
        }
        LOG(INFO)<<"globalaveragepool compute end";
    }
}
#endif