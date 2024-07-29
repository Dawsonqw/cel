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

        int output_h = (height + pad_top+pad_bottom - (dilation_h * (kernel_height - 1) + 1)) / stride_h + 1;
        int output_w = (width + pad_left+pad_right - (dilation_w * (kernel_width - 1) + 1)) / stride_w + 1;

        int32_t output_height=output_h*output_w;
        int32_t output_width=channel*kernel_height*kernel_width;

        // weight
        LOG_IF(FATAL,kernel.size()<1)<<"kernel size must be greater than 1";
        int32_t kernel_batch=kernel.size();
        int32_t kernel_channel=kernel[0]->channels();
        std::shared_ptr<Tensor<T>> kernel_matrix=std::make_shared<Tensor<T>>(kernel_channel*kernel_height*kernel_width,kernel_batch);
        for(int32_t index=0;index<kernel_batch;index++){
            std::shared_ptr<Tensor<T>> kernel_tensor=kernel[index];
            T* data=kernel_tensor->raw_data();
            T* kernel_cube=kernel_matrix->raw_data()+index*kernel_channel*kernel_height*kernel_width;
            memcpy(kernel_cube,data,kernel_channel*kernel_height*kernel_width*sizeof(T));
        }

        std::shared_ptr<cel::Tensor<T>> output_tensor=std::make_shared<cel::Tensor<T>>(output_height,output_width);
        output_tensor->Fill(static_cast<T>(0));
        for(int32_t index=0;index<batch;index++){
            std::shared_ptr<Tensor<T>> input_tensor=input[index];
            im2col(input_tensor,output_tensor,index,pads,stride,dilation,kernel_shape);
            std::shared_ptr<Tensor<T>> output_matrix=std::make_shared<Tensor<T>>();
            output_matrix=cel::matmul(output_tensor,kernel_matrix);
            output_matrix=cel::add(output_matrix,bias[0]);
            output_matrix->Reshape({output_h,output_w,channel});
            output_matrix->Transpose({2,0,1}); 
            output.push_back(output_matrix);
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

        int pad_top = pads[0];
        int pad_bottom = pads[1];
        int pad_left = pads[2];
        int pad_right = pads[3];
        int stride_h = stride[0];
        int stride_w = stride[1];
        int dil_h = dilation[0];
        int dil_w = dilation[1];
        int kernel_h = kernel_shape[0];
        int kernel_w = kernel_shape[1];

        int output_h = (rows + pad_top+pad_bottom - (dil_h * (kernel_h - 1) + 1)) / stride_h + 1;
        int output_w = (cols + pad_left+pad_right - (dil_w * (kernel_w - 1) + 1)) / stride_w + 1;

        int32_t output_height=output_h*output_w;
        int32_t output_width=channels*kernel_h*kernel_w;

        for(int32_t h_index=0;h_index<output_height;h_index++){
            for(int32_t w_index=0;w_index<output_width;w_index++){
                int32_t cur_index=w_index/(kernel_h*kernel_w);
                int32_t cur_x=h_index%output_w;
                int32_t cur_y=h_index/output_w;
                int32_t kernel_h_index=(w_index-cur_index*kernel_h*kernel_w)/kernel_w;
                int32_t kernel_w_index=(w_index-cur_index*kernel_h*kernel_w)%kernel_w;
                int32_t row_index=cur_y*stride_h+kernel_h_index*dil_h-pad_top;
                int32_t col_index=cur_x*stride_w+kernel_w_index*dil_w-pad_left;
                if (row_index<0||row_index>=rows||col_index<0||col_index>=cols){
                    output->set_data(h_index,w_index,0);
                }else{
                    output->set_data(h_index,w_index,input->at(cur_index,row_index,col_index));
                }
            }
        }
    }

    template<typename T>
    void col2im(cel::tensor_vec<T> &input,const std::vector<int64_t>& input_shape,cel::tensor_vec<T>&output,
                const std::vector<int64_t>&pads,const std::vector<int64_t>&stride,const std::vector<int64_t>&dilation,
                const std::vector<int64_t>&kernel_shape){
        int32_t batch_size=input_shape[0];
        int32_t channels=input_shape[1];
        int32_t rows=input_shape[2];
        int32_t cols=input_shape[3];

        int pad_top = pads[0];
        int pad_bottom = pads[1];
        int pad_left = pads[2];
        int pad_right = pads[3];
        int stride_h = stride[0];
        int stride_w = stride[1];
        int dil_h = dilation[0];
        int dil_w = dilation[1];
        int kernel_h = kernel_shape[0];
        int kernel_w = kernel_shape[1];

        int output_h = (rows + pad_top+pad_bottom- (dil_h * (kernel_h - 1) + 1)) / stride_h + 1;
        int output_w = (cols + pad_left+pad_right - (dil_w * (kernel_w - 1) + 1)) / stride_w + 1;

        int32_t output_height=output_h*output_w;
        int32_t output_width=channels*kernel_h*kernel_w;

        output.resize(batch_size);
        for (int32_t index=0;index<batch_size;index++){
            std::shared_ptr<Tensor<T>> output_tensor=std::make_shared<Tensor<T>>();
            output_tensor->set_size({channels,rows,cols});
            output[index]=output_tensor;
            for(int32_t h_index=0;h_index<output_height;h_index++){
                for(int32_t w_index=0;w_index<output_width;w_index++){
                    int32_t cur_index=w_index/(kernel_h*kernel_w);
                    int32_t cur_x=h_index%output_w;
                    int32_t cur_y=h_index/output_w;
                    int32_t kernel_h=(w_index-cur_index*kernel_h*kernel_w)/kernel_w;
                    int32_t kernel_w=(w_index-cur_index*kernel_h*kernel_w)%kernel_w;
                    int32_t row_index=cur_y*stride_h+kernel_h*dil_h-pad_top;
                    int32_t col_index=cur_x*stride_w+kernel_w*dil_w-pad_left;
                    if(row_index>=0&&row_index<rows&&col_index>=0&&col_index<cols){
                        output[index]->set_data(cur_index,row_index,col_index,input[index]->at(h_index,w_index));
                    }
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