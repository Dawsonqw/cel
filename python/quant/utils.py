import numpy as np

def im2col(input: np.array, kernel_shapes: np.array, strides: np.array, paddings: np.array, dilation: np.array):
    """
    将输入的多维数组转换为二维数组
    :param input: 输入的多维数组
    :param kernel_shapes: 卷积核的形状
    :param strides: 步长
    :param paddings: 填充
    :param dilation: 膨胀
    :return: 转换后的二维数组
    """
    batch_size,channels, height, width = input.shape
    kernel_height, kernel_width = kernel_shapes
    stride_height, stride_width = strides
    padding_height_top, padding_height_bottom, padding_width_left, padding_width_right = paddings
    dilation_height, dilation_width = dilation
    output_height = (height + padding_height_bottom+padding_height_bottom - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    output_width = (width + padding_width_left+padding_width_right- dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    output_h=output_height*output_width
    output_w=kernel_height*kernel_width*channels
    
    output=np.zeros((batch_size,output_h,output_w))

    for i in range(batch_size):
        for j in range(output_h):
            for k in range(output_w):
                c=k//(kernel_height*kernel_width)
                x=j%output_width
                y=j//output_width
                c=k//(kernel_height*kernel_width)
                k_i=(k-c*kernel_height*kernel_width)//kernel_width
                k_j=(k-c*kernel_height*kernel_width)%kernel_width
                row= y*stride_height+k_i*dilation_height-padding_height_top
                col= x*stride_width+k_j*dilation_width-padding_width_left
                if row<0 or row>=height or col<0 or col>=width:
                    output[i,j,k]=0
                else:
                    output[i,j,k]=input[i,c,row,col]

    output=output.reshape(batch_size*output_h,output_w)
    return output

def col2im(ori_input: np.array, input_shape: tuple, kernel_shapes: np.array, strides: np.array, paddings: np.array, dilation: np.array):
    """
    将二维数组转换回原来的多维数组
    :param ori_input: 二维数组 (batch_size * output_h, output_w)
    :param input_shape: 原始输入张量的形状 (batch_size, channels, height, width)
    :param kernel_shapes: 卷积核的形状 (kernel_height, kernel_width)
    :param strides: 步长 (stride_height, stride_width)
    :param paddings: 填充 (padding_height, padding_width)
    :param dilation: 膨胀 (dilation_height, dilation_width)
    :return: 转换后的多维数组 (batch_size, channels, height, width)
    """
    batch_size, channels, height, width = input_shape
    kernel_height, kernel_width = kernel_shapes
    stride_height, stride_width = strides
    padding_height_top, padding_height_bottom, padding_width_left, padding_width_right = paddings
    dilation_height, dilation_width = dilation

    output_height = (height +padding_height_bottom+padding_height_top - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    output_width = (width + padding_width_left+padding_width_right - dilation_width * (kernel_width - 1) - 1) // stride_width + 1

    output_h = output_height * output_width
    output_w = kernel_height * kernel_width * channels

    padded_input = np.zeros((batch_size, channels, height, width))
    ori_input = ori_input.reshape(batch_size, output_h, output_w)

    for i in range(batch_size):
        for j in range(output_h):
            for k in range(output_w):
                c=k//(kernel_height*kernel_width)
                x=j%output_width
                y=j//output_width
                c=k//(kernel_height*kernel_width)
                k_i=(k-c*kernel_height*kernel_width)//kernel_width
                k_j=(k-c*kernel_height*kernel_width)%kernel_width
                row= y*stride_height+k_i*dilation_height-padding_height_top
                col= x*stride_width+k_j*dilation_width-padding_width_left
                if row>=0 and row<height and col>=0 and col<width:
                    padded_input[i, c, row, col] = ori_input[i, j, k]
    return padded_input

def Conv(input: np.array, weight: np.array, bias: np.array, kernel_shapes: np.array, strides: np.array, paddings: np.array, dilation: np.array):
    """
    卷积操作
    :param input: 输入张量
    :param weight: 权重张量
    :param bias: 偏置张量
    :param kernel_shapes: 卷积核的形状
    :param strides: 步长
    :param paddings: 填充
    :param dilation: 膨胀
    :return: 输出张量
    """
    col_input = im2col(input, kernel_shapes, strides, paddings, dilation)
    print("col shape:",col_input.shape)
    col_weight = weight.reshape(weight.shape[0], -1).T
    print("col_weight shape:",col_weight.shape)
    output=np.dot(col_input,col_weight)+bias
    output_batch_size = input.shape[0]
    output_channels = weight.shape[0]
    output_height = (input.shape[2] + paddings[1]+paddings[0] - dilation[0] * (kernel_shapes[0] - 1) - 1) // strides[0] + 1
    output_width = (input.shape[3] + paddings[2]+paddings[3] - dilation[1] * (kernel_shapes[1] - 1) - 1) // strides[1] + 1
    output=output.reshape(output_batch_size,output_height,output_width,output_channels).transpose(0,3,1,2)
    return output


input=np.random.randn(1,128,28,28).astype(np.float32)
weight=np.random.randn(128,128,3,3).astype(np.float32)
bias=np.random.randn(128).astype(np.float32)
kernel_shapes=np.array([weight.shape[2],weight.shape[3]])
strides=np.array([1,1])
paddings=np.array([1,1,1,1])
dilation=np.array([1,1])
output=Conv(input,weight,bias,kernel_shapes,strides,paddings,dilation)
print("output:",output.shape)

import torch
input_torch=torch.tensor(input)
weight_torch=torch.tensor(weight)
bias_torch=torch.tensor(bias)
output_torch=torch.nn.functional.conv2d(input_torch,weight_torch,bias=bias_torch,padding=[1,1],\
                                        stride=strides.tolist(),dilation=dilation.tolist())
print("output_torch:",output_torch.shape)
print("diff:",np.sum(np.abs(output-output_torch.numpy())))
print(np.allclose(output,output_torch.numpy(),atol=1e-4))
# print(output==output_torch.numpy())
print("="*40)
print(output-output_torch.numpy())
print("="*40)