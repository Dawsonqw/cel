#ifndef PARSER_COMPUTE_CONV_H
#define PARSER_COMPUTE_CONV_H
#include "utils.hpp"
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"
#include "Parser/tensor_utils.hpp"

namespace cel{

    template<typename T>
    void conv_compute(const tensor_vec<T>&input,const tensor_vec<T>&kernel,const tensor_vec<T>& bias,tensor_vec<T>&output,
                    const std::vector<int64_t>&pads,const std::vector<int64_t>&stride,const std::vector<int64_t>&dilation,
                    const std::vector<int64_t>&kernel_shape,int64_t group=1){
        int32_t batch=input.size();
        LOG_IF(FATAL,batch<1)<<"input size must be greater than 1";
        std::vector<int32_t> input_shape=input[0]->raw_shapes();
        LOG_IF(FATAL,input_shape.size()!=3)<<"input shape must be 3";
        int32_t channel=input_shape[0];
        int32_t height=input_shape[1];
        int32_t width=input_shape[2];
    
        int64_t kernel_width=kernel_shape[0];
        int64_t kernel_height=kernel_shape[1];

        int64_t pad_top=pads[0];
        int64_t pad_bottom=pads[1];
        int64_t pad_left=pads[2];
        int64_t pad_right=pads[3];

        int64_t stride_h=stride[0];
        int64_t stride_w=stride[1];

        int64_t dilation_h=dilation[0];
        int64_t dilation_w=dilation[1];

        int output_h = (height + pad_top+pad_bottom - (dilation_h * (kernel_height - 1) + 1)) / stride_h + 1;
        int output_w = (width + pad_left+pad_right - (dilation_w * (kernel_width - 1) + 1)) / stride_w + 1;

        int32_t output_height=output_h*output_w;
        int32_t output_width=channel*kernel_height*kernel_width;

        LOG_IF(FATAL,kernel.size()<1)<<"kernel size must be greater than 1";
        int32_t kernel_batch=kernel.size();
        int32_t kernel_channel=kernel[0]->channels();
        std::shared_ptr<Tensor<T>> kernel_matrix=std::make_shared<Tensor<T>>(kernel_channel*kernel_height*kernel_width,kernel_batch);
        for(int32_t index=0;index<kernel_batch;index++){
            std::shared_ptr<Tensor<T>> kernel_tensor=kernel[index];
            memcpy(kernel_matrix->raw_ptr(index*kernel_channel*kernel_height*kernel_width),kernel_tensor->raw_ptr(),kernel_channel*kernel_height*kernel_width*sizeof(T));

        }

        std::shared_ptr<cel::Tensor<T>> output_tensor=std::make_shared<cel::Tensor<T>>(output_height,output_width);
        output_tensor->Fill(static_cast<T>(0));
        for(int32_t index=0;index<batch;index++){
            std::shared_ptr<Tensor<T>> input_tensor=input[index];
            im2col(input_tensor,output_tensor,pads,stride,dilation,kernel_shape);
            std::shared_ptr<Tensor<T>> output_matrix=std::make_shared<Tensor<T>>();
            output_matrix=cel::matmul(output_tensor,kernel_matrix);
            output_matrix=cel::add(output_matrix,bias[0]);
            output_matrix->Reshape({output_h,output_w,kernel_batch});
            output_matrix->Permute({2,0,1}); 
            output.push_back(output_matrix);
        }

        LOG(INFO)<<"conv compute done";
    }


    void conv_shape_infer(const std::vector<int32_t>&input_shape,const std::vector<int64_t>&kernel_shape,
                        const std::vector<int64_t>&pads,const std::vector<int64_t>&stride,const std::vector<int64_t>&dilation,
                        std::vector<int64_t>&output_shape,int64_t group=1){
        int32_t batch=input_shape[0];
        int32_t channel=input_shape[1];
        int32_t height=input_shape[2];
        int32_t width=input_shape[3];

        int64_t kernel_channel=kernel_shape[0];
        int64_t kernel_height=kernel_shape[1];
        int64_t kernel_width=kernel_shape[2];

        int64_t pad_top=pads[0];
        int64_t pad_bottom=pads[1];
        int64_t pad_left=pads[2];
        int64_t pad_right=pads[3];

        int64_t stride_h=stride[0];
        int64_t stride_w=stride[1];

        int64_t dilation_h=dilation[0];
        int64_t dilation_w=dilation[1];

        int64_t output_channel=kernel_shape[0];
        int64_t output_height=(height+pad_top+pad_bottom-(dilation_h*(kernel_height-1)+1))/stride_h+1;
        int64_t output_width=(width+pad_left+pad_right-(dilation_w*(kernel_width-1)+1))/stride_w+1;

        output_shape={batch,output_channel,output_height,output_width};
    }
}
#endif