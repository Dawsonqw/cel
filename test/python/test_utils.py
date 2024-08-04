import numpy as np
import torch
from quant.utils import im2col,col2im,im2col_opt,col2im_opt


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
    col_weight = weight.reshape(weight.shape[0], -1).T
    output=np.dot(col_input,col_weight)+bias
    output_batch_size = input.shape[0]
    output_channels = weight.shape[0]
    output_height = (input.shape[2] + paddings[1]+paddings[0] - dilation[0] * (kernel_shapes[0] - 1) - 1) // strides[0] + 1
    output_width = (input.shape[3] + paddings[2]+paddings[3] - dilation[1] * (kernel_shapes[1] - 1) - 1) // strides[1] + 1
    output=output.reshape(output_batch_size,output_height,output_width,output_channels).transpose(0,3,1,2)
    print("output_height:",output_height,"output_width:",output_width,"output_channels:",output_channels)
    return output

# 实现MaxPool2D 利用im2col实现 存在ceil_mode,需要进行判断，也有dilations
def MaxPool2D(input: np.array, kernel_shapes: np.array, strides: np.array, paddings: np.array, dilations: np.array, ceil_mode: int):
    """
    最大池化操作
    :param input: 输入张量
    :param kernel_shapes: 池化核的形状
    :param strides: 步长
    :param paddings: 填充
    :param dilations: 膨胀
    :param ceil_mode: 是否向上取整
    :return: 输出张量
    """
    batch_size, channels, height, width = input.shape
    kernel_height, kernel_width = kernel_shapes
    stride_height, stride_width = strides
    padding_height_top, padding_height_bottom, padding_width_left, padding_width_right = paddings
    dilation_height, dilation_width = dilations

    output_height = (height + padding_height_bottom+padding_height_top - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    output_width = (width + padding_width_left+padding_width_right - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    output = np.zeros((batch_size, channels, output_height, output_width))

    for i in range(batch_size):
        for j in range(channels):
            for k in range(output_height):
                for l in range(output_width):
                    row_start = k * stride_height - padding_height_top
                    row_end = row_start + kernel_height
                    col_start = l * stride_width - padding_width_left
                    col_end = col_start + kernel_width
                    row_start = max(row_start, 0)
                    row_end = min(row_end, height)
                    col_start = max(col_start, 0)
                    col_end = min(col_end, width)
                    pool_region = input[i, j, row_start:row_end, col_start:col_end]
                    if ceil_mode:
                        output[i, j, k, l] = np.max(pool_region)
                    else:
                        output[i, j, k, l] = np.max(pool_region)
    return output


def test_for_conv():
    input=np.random.randn(1,3,224,224).astype(np.float32)
    weight=np.random.randn(64,3,7,7).astype(np.float32)
    bias=np.random.randn(64).astype(np.float32)
    kernel_shapes=np.array([weight.shape[2],weight.shape[3]])
    strides=np.array([2,2])
    paddings=np.array([3,3,3,3])
    dilation=np.array([1,1])
    output=Conv(input,weight,bias,kernel_shapes,strides,paddings,dilation)
    print("output:",output.shape)

    input=np.random.randn(1,3,224,224).astype(np.float32)
    weight=np.random.randn(64,3,7,7).astype(np.float32)
    bias=np.random.randn(64).astype(np.float32)
    input_torch=torch.tensor(input)
    weight_torch=torch.tensor(weight)
    bias_torch=torch.tensor(bias)
    output_torch=torch.nn.functional.conv2d(input_torch,weight_torch,bias=bias_torch,padding=[3,3],\
                                            stride=strides.tolist(),dilation=dilation.tolist())
    print("output_torch:",output_torch.shape)
    print("diff:",np.sum(np.abs(output-output_torch.numpy())))
    print(np.allclose(output,output_torch.numpy(),atol=1e-4))


def test_for_maxpool():
    # 测试MaxPool
    input=np.random.randn(1,64,112,112).astype(np.float32)
    ceil_mode=0
    kernel_shapes=np.array([3,3])
    dilations=np.array([1,1])
    paddings=np.array([1,1,1,1])
    strides=np.array([2,2])
    output=MaxPool2D(input,kernel_shapes,strides,paddings,dilations,ceil_mode)

    # torch maxpool
    input_torch=torch.tensor(input)
    output_torch=torch.nn.functional.max_pool2d(input_torch,kernel_size=3,stride=2,padding=1)
    print("output_torch:",output_torch.shape)
    print("diff:",np.sum(np.abs(output-output_torch.numpy())))
    print(np.allclose(output,output_torch.numpy(),atol=1e-4))

def test_for_im2col_col2im():
    input=np.random.randn(1,64,56,56).astype(np.float32)
    kernel_shapes=np.array([1,1])
    strides=np.array([2,2])
    paddings=np.array([0,0,0,0])
    dilation=np.array([1,1])
    col_input=im2col(input,kernel_shapes,strides,paddings,dilation)

    # col_opt_input=im2col_opt(input,7,7,3,3,3,3,1,1,2,2)
    col_opt_input=im2col_opt(input,kernel_shapes[0],kernel_shapes[1],paddings[0],paddings[1],paddings[2],paddings[3],
                             dilation[0],dilation[1],strides[0],strides[1])
    # col2im_opt_input=col2im_opt(col_opt_input,input.shape,7,7,2,1,1,2,3,3,3,3)
    col2im_opt_input=col2im_opt(col_opt_input,input.shape,kernel_shapes[0],kernel_shapes[1],
                                paddings[0],paddings[1],paddings[2],paddings[3],dilation[0],dilation[1],strides[0],strides[1])

    print("col_input:",col_input.shape)
    input_shape=input.shape
    col2im_input=col2im(col_input,input_shape,kernel_shapes,strides,paddings,dilation)
    print("col2im_input:",col2im_input.shape)
    print(np.allclose(input,col2im_input,atol=1e-4))
    print("col2im_opt_input:",col2im_opt_input.shape)
    print(np.allclose(input,col2im_opt_input,atol=1e-4))
    print("=====================")
    # print(input)
    print("=====================")
    # print(col2im_input)
    print("=====================")

# test_for_conv()
# test_for_maxpool()
test_for_im2col_col2im()