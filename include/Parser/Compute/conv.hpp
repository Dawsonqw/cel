#ifndef PARSER_COMPUTE_CONV_H
#define PARSER_COMPUTE_CONV_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"

namespace cel{

    // 目前仅支持NCHW的数据排布
    template<typename T>
    void conv_compute(const tensor_vec<T>&input,const tensor_vec<T>&kernel,const tensor_vec<T>& bias,tensor_vec<T>&output,
                    const std::vector<int32_t>&padding,const std::vector<int32_t>&stride,const std::vector<int32_t>&dilation,
                    const std::vector<int32_t>&kernel_shape,int64_t group=1){

        int32_t batch=input.size();
        std::vector<int32_t> input_shape=input[0]->shape();
        int32_t channel=input_shape[0];
        int32_t height=input_shape[1];
        int32_t width=input_shape[2];

        int32_t kernel_channel=kernel_shape[0];
        int32_t kernel_height=kernel_shape[1];

        int32_t pad_top=padding[0];
        int32_t pad_bottom=padding[1];
        int32_t pad_left=padding[2];
        int32_t pad_right=padding[3];

        int32_t stride_h=stride[0];
        int32_t stride_w=stride[1];

        int32_t dilation_h=dilation[0];
        int32_t dilation_w=dilation[1];

        for(int32_t batch_id=0;batch_id<batch;batch_id++){
            for(int32_t group_id=0;group_id<group;group_id++){
                im2col();
            }
        }

        
    }

    template<typename T>
    void im2col(
    ){

    }

    void conv_shape_infer(const std::vector<int32_t>&input_shape,const std::vector<int32_t>&kernel_shape,
                        const std::vector<int32_t>&padding,const std::vector<int32_t>&stride,const std::vector<int32_t>&dilation,
                        std::vector<int32_t>&output_shape,int64_t group=1){
        int32_t batch=input_shape[0];
        int32_t channel=input_shape[1];
        int32_t height=input_shape[2];
        int32_t width=input_shape[3];

        int32_t kernel_channel=kernel_shape[0];
        int32_t kernel_height=kernel_shape[1];
        int32_t kernel_width=kernel_shape[2];

        int32_t pad_top=padding[0];
        int32_t pad_bottom=padding[1];
        int32_t pad_left=padding[2];
        int32_t pad_right=padding[3];

        int32_t stride_h=stride[0];
        int32_t stride_w=stride[1];

        int32_t dilation_h=dilation[0];
        int32_t dilation_w=dilation[1];

        int32_t output_channel=kernel_shape[0];
        int32_t output_height=(height+pad_top+pad_bottom-(dilation_h*(kernel_height-1)+1))/stride_h+1;
        int32_t output_width=(width+pad_left+pad_right-(dilation_w*(kernel_width-1)+1))/stride_w+1;

        output_shape={batch,output_channel,output_height,output_width};
    }
}
#endif