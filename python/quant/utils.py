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
    padding_height_top,padding_width_left, padding_height_bottom, padding_width_right = paddings
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
    padding_height_top,padding_width_left, padding_height_bottom, padding_width_right = paddings
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

def im2col_opt(input_data, kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, dilation_h, dilation_w, stride_h, stride_w):
    N, C, H, W = input_data.shape
    out_h = (H + pad_t + pad_b - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (W + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    img = np.pad(input_data, ((0,0), (0,0), (pad_t, pad_b), (pad_l, pad_r)), 'constant')
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))

    for y in range(kernel_h):
        y_max = y + stride_h * out_h
        for x in range(kernel_w):
            x_max = x + stride_w * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride_h, x:x_max:stride_w]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im_opt(col, input_shape, kernel_h, kernel_w,pad_t,pad_l,pad_b,pad_r,dilation_h,dilation_w,stride_h,stride_w):
    N, C, H, W = input_shape
    out_h = (H + pad_t + pad_b - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (W + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    col=col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    img=np.zeros((N, C, H + pad_t + pad_b, W + pad_l + pad_r))
    for y in range(kernel_h):
        y_max = y + stride_h * out_h
        for x in range(kernel_w):
            x_max = x + stride_w * out_w
            img[:, :, y:y_max:stride_h, x:x_max:stride_w] = col[:, :, y, x, :, :]
    return img[:, :, pad_t:H + pad_t, pad_l:W + pad_l]

