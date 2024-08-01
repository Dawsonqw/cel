import numpy as np
import os
import torch

def conv2d_infer(input,weight,bias,stride,padding,dilation,groups):
    return torch.nn.functional.conv2d(input,weight,bias,stride,padding,dilation,groups)

def test_conv2d_infer():
    print("start test conv2d_infer")
    input_name="/root/workspace/cel/build/conv_input.bin"
    weight_name="/root/workspace/cel/build/conv_weight.bin"
    bias_name="/root/workspace/cel/build/conv_bias.bin"
    output_name="/root/workspace/cel/build/conv_output.bin"
    stride=[1,1]
    padding=[1,1]
    dilation=[1,1]
    groups=1
    input_data=torch.tensor(np.random.randn(1,64,56,56).astype(np.float32))
    weight_data=torch.tensor(np.random.randn(64,64,3,3).astype(np.float32))
    bias_data=torch.tensor(np.random.randn(64).astype(np.float32))
    output=conv2d_infer(input_data,weight_data,bias_data,stride,padding,dilation,groups)
    output_data=output.data.numpy()
    input_data=input_data.data.numpy()
    weight_data=weight_data.data.numpy()
    bias_data=bias_data.data.numpy()
    input_data.tofile(input_name)
    weight_data.tofile(weight_name)
    bias_data.tofile(bias_name)
    output_data.tofile(output_name)
    print("done")
    
if __name__ == "__main__":
    test_conv2d_infer()
        