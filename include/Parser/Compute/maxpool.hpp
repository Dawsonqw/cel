#ifndef PARSER_COMPUTE_MAXPOOL_H
#define PARSER_COMPUTE_MAXPOOL_H
#include "utils.hpp"
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"
#include "Parser/tensor_utils.hpp"

namespace cel{

    template<typename T>
    void maxpool_compute(const tensor_vec<T>&input,tensor_vec<T>&output,int64_t ceil_mode,std::vector<int64_t> kernel_shape,std::vector<int64_t> pads,
                                            std::vector<int64_t> strides,std::vector<int64_t> dilations){
        LOG(INFO)<<"maxpool compute start";
        int32_t batch_size=input.size();
        LOG_IF(FATAL,batch_size<1)<<"input size must be greater than 1";
        std::vector<int32_t> input_shape=input[0]->raw_shapes();
        LOG_IF(FATAL,input_shape.size()!=3)<<"input shape must be 3";
        int32_t channels=input_shape[0];
        int32_t height=input_shape[1];
        int32_t width=input_shape[2];

        int32_t kernel_height=kernel_shape[0];
        int32_t kernel_width=kernel_shape[1];

        int32_t stride_height=strides[0];
        int32_t stride_width=strides[1];

        int32_t pad_top=pads[0];
        int32_t pad_bottom=pads[1];
        int32_t pad_left=pads[2];
        int32_t pad_right=pads[3];

        int32_t dilation_height=dilations[0];
        int32_t dilation_width=dilations[1];

        int32_t output_height=(height+pad_top+pad_bottom-(dilation_height*(kernel_height-1)+1))/stride_height+1;
        int32_t output_width=(width+pad_left+pad_right-(dilation_width*(kernel_width-1)+1))/stride_width+1;

        output.resize(batch_size);
        for(int32_t index=0;index<batch_size;index++){
            std::shared_ptr<Tensor<T>> input_tensor=input[index];
            std::shared_ptr<Tensor<T>> output_tensor=std::make_shared<Tensor<T>>();
            output_tensor->set_size({channels,output_height,output_width});
            for(int32_t c=0;c<channels;c++){
                for(int32_t h=0;h<output_height;h++){
                    for(int32_t w=0;w<output_width;w++){
                        int32_t h_start=h*stride_height-pad_top;
                        int32_t h_end=h_start+kernel_height*dilation_height;
                        int32_t w_start=w*stride_width-pad_left;
                        int32_t w_end=w_start+kernel_width*dilation_width;
                        h_start=std::max(h_start,0);
                        h_end=std::min(h_end,height);
                        w_start=std::max(w_start,0);
                        w_end=std::min(w_end,width);
                        T max_value=std::numeric_limits<T>::lowest();
                        for(int32_t h_index=h_start;h_index<h_end;h_index+=dilation_height){
                            for(int32_t w_index=w_start;w_index<w_end;w_index+=dilation_width){
                                max_value=std::max(max_value,input_tensor->at(c,h_index,w_index));
                            }
                        }
                        output_tensor->at(c,h,w)=max_value;
                    }
                }
            }
            output[index]=output_tensor;
        }
    
    }
}
#endif