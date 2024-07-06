#ifndef PARSER_COMPUTE_CONV_H
#define PARSER_COMPUTE_CONV_H
#include "Parser/utils.hpp"
#include "Parser/tensor.hpp"

namespace cel{

    template<typename T>
    void conv_compute(const tensor_vec<T>&input,const tensor_vec<T>&kernel,const tensor_vec<T>& bias,tensor_vec<T>&output,
                    const std::vector<int64_t>&pads,const std::vector<int64_t>&stride,const std::vector<int64_t>&dilation,
                    const std::vector<int64_t>&kernel_shape,int64_t group=1){
        int32_t batch=input.size();
        // std::vector<int32_t> input_shape=input[0]->shapes();
        // int32_t channel=input_shape[0];
        // int32_t height=input_shape[1];
        // int32_t width=input_shape[2];

        // int64_t kernel_channel=kernel_shape[0];
        // int64_t kernel_height=kernel_shape[1];

        // int64_t pad_top=pads[0];
        // int64_t pad_bottom=pads[1];
        // int64_t pad_left=pads[2];
        // int64_t pad_right=pads[3];

        // int64_t stride_h=stride[0];
        // int64_t stride_w=stride[1];

        // int64_t dilation_h=dilation[0];
        // int64_t dilation_w=dilation[1];

        // output.resize(batch);

        // for(int64_t batch_id=0;batch_id<batch;batch_id++){
        //     for(int64_t group_id=0;group_id<group;group_id++){
        //         std::shared_ptr<Tensor<T>> input_tensor=input[batch_id];
        //         std::shared_ptr<Tensor<T>> input_col(new Tensor<T>());
        //         im2col<T>(input_tensor,input_col,pads,stride,dilation,kernel_shape);
        //         std::shared_ptr<Tensor<T>> output_tensor(new Tensor<T>());
        //         output_tensor->set_size({kernel_channel,height,width});
        //         output_tensor->Fill(0.0f);
        //         for(int64_t kernel_id=0;kernel_id<kernel_channel;kernel_id++){
        //             std::shared_ptr<Tensor<T>> kernel_tensor=kernel[group_id*kernel_channel+kernel_id];
        //             // std::shared_ptr<Tensor<T>> bias_tensor=bias[group_id*kernel_channel+kernel_id];
        //             // std::shared_ptr<Tensor<T>> output_col(new Tensor<T>());
        //             // output_col->set_size({height,width});
        //             // output_col->Fill(0.0f);
        //             // output_tensor->data().slice(kernel_id)=kernel_tensor->data()*input_col->data();
        //             // output_tensor->data().slice(kernel_id)+=bias_tensor->data();
        //         }
        //         output[batch_id]=output_tensor;
        //     }
        // }
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

        // 每次卷积的结果是一个kernel_h*kernel_w的矩阵，一共channels*kernel_h*kernel_w个矩阵
        // 停留次数是output_h*output_w,即w方向移动output_w次，h方向移动output_h次
        // 由于是列主序，所以将行数设为channels*kernel_h*kernel_w，列数设为output_h*output_w
        int32_t col_h=channels*kernel_h*kernel_w;
        int32_t col_w=output_h*output_w;
        output->set_size({col_h,col_w});

        for(int32_t h_index=0;h_index<col_h;h_index++){
            for(int32_t w_index=0;w_index<col_w;w_index++){
            }
        }

        // 使用set_data设置数据，用at获取数据
        // for(int32_t channel_index=0;channel_index<channels;channel_index++){
        //     for(int32_t move_w=0;move_w<output_w;move_w++){
        //         for(int32_t move_h=0;move_h<output_h;move_h++){
        //             int32_t h_start = move_h * stride_h - pad_h;
        //             int32_t w_start = move_w * stride_w - pad_w;
        //             int32_t h_end=h_start+kernel_h*dil_h;
        //             int32_t w_end=w_start+kernel_w*dil_w;
        //             int32_t output_num=output_w*move_h+move_w;
        //             for(int32_t h=h_start;h<h_end;h+=dil_h){
        //                 for(int32_t w=w_start;w<w_end;w+=dil_w){
        //                     int32_t output_row=channel_index*kernel_h*kernel_w+(h-h_start)/dil_h*kernel_w+(w-w_start)/dil_w;
        //                     if(h>=0&&h<rows&&w>=0&&w<cols){
        //                         output->set_data(output_row,output_num,input->at(channel_index,h,w));
        //                     }else{
        //                         output->set_data(output_row,output_num,0);
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

        // for(int c=0;c<channels;c++){
        //     for(int kh=0;kh<kernel_h;kh++){
        //         for(int kw=0;kw<kernel_w;kw++){
        //             int input_h_offset=c*rows*cols;
        //             int output_h_offset=(c*kernel_h*kernel_w+kh*kernel_w+kw)*output_h*output_w;

        //             for(int oh=0;oh<output_h;oh++){
        //                 for(int ow=0;ow<output_w;ow++){
        //                     int ih=oh*stride_h-pad_h+kh*dil_h;
        //                     int iw=ow*stride_w-pad_w+kw*dil_w;

        //                     if(ih>=0&&ih<rows&&iw>=0&&iw<cols){
        //                         output->data()[output_h_offset+oh*output_w+ow]=input->data()[input_h_offset+ih*cols+iw];
        //                     }else{
        //                         output->data()[output_h_offset+oh*output_w+ow]=0;
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }

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