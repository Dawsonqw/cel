import numpy as np
from quant.awq import quantize
from quant.utils import im2col,col2im


def test_quantize():
    input=np.random.rand(128,64).astype(np.float32)
    weight=np.random.rand(64,64).astype(np.float32)
    
    real_output=np.random.rand(8,2).astype(np.float32)
    group_size=64
    max_chunk_memory=1024*1024*1024
    
    for_construct_mul_op,afer_update_weight=quantize(input,weight,real_output,group_size,max_chunk_memory,True)

def test_conv_quantize():
    input=np.random.rand(1,3,224,224).astype(np.float32)
    weight=np.random.rand(64,3,7,7).astype(np.float32)
    dilation=np.array([1,1])
    paddings=np.array([3,3,3,3])
    strides=np.array([2,2])
    
    input_feat=im2col(input,(7,7),strides,paddings,dilation)
    weight_feat=weight.reshape(weight.shape[0],-1).T
    print("input shape:",input_feat.shape)
    print("weight shape:",weight_feat.shape)
    output=np.dot(input_feat,weight_feat)

    group_size=weight_feat.shape[0]
    max_chunk_memory=1024*1024*1024
    do_clip=True
    after_update_inputs,afer_update_weight=quantize(input_feat,weight_feat,output,group_size,max_chunk_memory,do_clip)

    after_output=np.dot(after_update_inputs,afer_update_weight)
    print("diff:",np.sum(np.abs(output-after_output)),np.acllclose(output,after_output,atol=1e-4))


# test_quantize()
test_conv_quantize()
    
