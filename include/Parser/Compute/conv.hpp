#ifndef PARSER_COMPUTE_CONV_H
#define PARSER_COMPUTE_CONV_H
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
        std::vector<int32_t> input_shape=input[0]->shapes();
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

        // input
        output.resize(batch);
        tensor_vec<T> input_matrix;
        input_matrix.resize(batch);
        for(int32_t index=0;index<batch;index++){
            std::shared_ptr<Tensor<T>> input_tensor=input[index];
            std::shared_ptr<Tensor<T>> output_tensor=std::make_shared<Tensor<T>>();
            im2col(input_tensor,output_tensor,pads,stride,dilation,kernel_shape);
            input_matrix[index]=output_tensor;
        }

        // weight
        int32_t kernel_batch=kernel.size();
        LOG_IF(FATAL,kernel.size()<1)<<kernel.size()<<" kernel size must be greater than 1";
        int32_t kernel_channel=kernel[0]->channels();
        std::shared_ptr<Tensor<T>> kernel_matrix=std::make_shared<Tensor<T>>();
        int32_t weight_height=kernel_channel*kernel_height*kernel_width;
        int32_t weight_width=kernel_batch;

        kernel_matrix->set_size({weight_height,weight_width});
        for(int32_t w_index=0;w_index<weight_width;w_index++){
            for(int32_t h_index=0;h_index<weight_height;h_index++){
                int32_t channel_index=h_index/(kernel_height*kernel_width);
                int32_t channel_offset=h_index%(kernel_height*kernel_width);
                int32_t h_offset=channel_offset/kernel_width;
                int32_t w_offset=channel_offset%kernel_width;
                kernel_matrix->set_data(h_index,w_index,kernel[w_index]->at(channel_index,h_offset,w_offset));
            }
        }

        tensor_vec<T> output_matrix;
        output_matrix.resize(batch);
        for(int32_t index=0;index<batch;index++){
            std::shared_ptr<Tensor<T>> input_matrix_tensor=input_matrix[index];
            std::shared_ptr<Tensor<T>> output_matrix_tensor=std::make_shared<Tensor<T>>();
            output_matrix_tensor->set_size({input_matrix_tensor->rows(),kernel_matrix->cols()});
            output_matrix_tensor=cel::matmul(input_matrix_tensor,kernel_matrix);
            output_matrix[index]=output_matrix_tensor;
        }

        LOG(INFO)<<"conv compute done";
    }


    template<typename T>
    void im2col(std::shared_ptr<Tensor<T>>&input,std::shared_ptr<Tensor<T>>&output,
                const std::vector<int64_t>&pads,const std::vector<int64_t>&stride,const std::vector<int64_t>&dilation,
                const std::vector<int64_t>&kernel_shape){
        int32_t channels=input->channels();
        int32_t rows=input->rows();
        int32_t cols=input->cols();

        int pad_h = pads[0];
        int pad_w = pads[1];
        int stride_h = stride[0];
        int stride_w = stride[1];
        int dil_h = dilation[0];
        int dil_w = dilation[1];
        int kernel_h = kernel_shape[0];
        int kernel_w = kernel_shape[1];

        int output_h = (rows + 2 * pad_h - (dil_h * (kernel_h - 1) + 1)) / stride_h + 1;
        int output_w = (cols + 2 * pad_w - (dil_w * (kernel_w - 1) + 1)) / stride_w + 1;

        int32_t output_height=output_h*output_w;
        int32_t output_width=channels*kernel_h*kernel_w;
        output->set_size({output_height,output_height});

        for(int32_t h_index=0;h_index<output_height;h_index++){
            for(int32_t w_index=0;w_index<output_width;w_index++){
                int32_t channel_index=w_index/(kernel_h*kernel_w);
                int32_t channel_offset=w_index%(kernel_h*kernel_w);
                int32_t h_offset=channel_offset/kernel_w;
                int32_t w_offset=channel_offset%kernel_w;
                int32_t h_start = h_offset * dil_h - pad_h;
                int32_t w_start = w_offset * dil_w - pad_w;
                int32_t h=h_start+w_index/output_w*stride_h;
                int32_t w=w_start+w_index%output_w*stride_w;
                if(h>=0&&h<rows&&w>=0&&w<cols){
                    output->set_data(h_index,w_index,input->at(channel_index,h,w));
                }else{
                    output->set_data(h_index,w_index,0);
                }
            }
        }
    }

    template<typename T>
    void col2im(std::shared_ptr<Tensor<T>>&input,std::shared_ptr<Tensor<T>>&output,
                const std::vector<int64_t>&pads,const std::vector<int64_t>&stride,const std::vector<int64_t>&dilation,
                const std::vector<int64_t>&kernel_shape){
        int32_t channels=output->channels();
        int32_t rows=output->rows();
        int32_t cols=output->cols();

        int pad_h = pads[0];
        int pad_w = pads[1];
        int stride_h = stride[0];
        int stride_w = stride[1];
        int dil_h = dilation[0];
        int dil_w = dilation[1];
        int kernel_h = kernel_shape[0];
        int kernel_w = kernel_shape[1];

        int output_h = (rows + 2 * pad_h - (dil_h * (kernel_h - 1) + 1)) / stride_h + 1;
        int output_w = (cols + 2 * pad_w - (dil_w * (kernel_w - 1) + 1)) / stride_w + 1;

        int32_t output_height=output_h*output_w;
        int32_t output_width=channels*kernel_h*kernel_w;

        for(int32_t h_index=0;h_index<output_height;h_index++){
            for(int32_t w_index=0;w_index<output_width;w_index++){
                int32_t channel_index=w_index/(kernel_h*kernel_w);
                int32_t channel_offset=w_index%(kernel_h*kernel_w);
                int32_t h_offset=channel_offset/kernel_w;
                int32_t w_offset=channel_offset%kernel_w;
                int32_t h_start = h_offset * dil_h - pad_h;
                int32_t w_start = w_offset * dil_w - pad_w;
                int32_t h=h_start+w_index/output_w*stride_h;
                int32_t w=w_start+w_index%output_w*stride_w;
                if(h>=0&&h<rows&&w>=0&&w<cols){
                    output->set_data(channel_index,h,w,input->at(h_index,w_index));
                }
            }
        }
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