import numpy as np
import onnx
import onnx.checker
import onnxruntime as ort

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

def col2im_opt(col, input_shape, kernel_h, kernel_w, stride_h,,stride_w,pad_l,pad_r,pad_t,pad_b):
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

import onnx
import onnxruntime as ort

def process(model_path,input_data):
    model=onnx.load(model_path)
    conv_dict_input={}
    conv_dict_input_data={}
    conv_dict_attr={}
    shape_info={}
    for node in model.graph.node:
        if node.op_type=="Conv":
            conv_dict_input[node.name]=node.input
            conv_dict_input_data[node.name]=node.input[0]
            conv_dict_attr[node.name]=node.attribute

    for item in model.graph.value_info:
        if item.name in conv_dict_input_data.values():
            shape=item.type.tensor_type.shape.dim
            shape=[dim.dim_value for dim in shape]
            shape_info[item.name]=shape


    for key in shape_info.keys():
        model.graph.output.extend([onnx.helper.make_tensor_value_info(key,onnx.TensorProto.FLOAT,shape_info[key])])

    ort_engine = ort.InferenceSession(model.SerializeToString())
    input_name = ort_engine.get_inputs()[0].name
    output_name = [out.name for out in ort_engine.get_outputs()]
    out_info={}
    for idx in range(len(output_name)):
        out_info[output_name[idx]]=idx
    result=ort_engine.run(output_name,{input_name:input_data})
    result_dict={}
    for key in out_info.keys():
        result_dict[key]=result[out_info[key]]
    return conv_dict_input,conv_dict_input_data,conv_dict_attr,result_dict,shape_info

def make_mul_op(node_name, input_name, weight,cst_idx,mul_idx,model,ins_node_name):
    tensor_weight=onnx.helper.make_tensor(name=node_name+"_cst",data_type=onnx.TensorProto.FLOAT,dims=weight.shape,vals=weight)
    node=onnx.helper.make_node("Constant",inputs=[],outputs=[node_name+"_cst_out"],value=tensor_weight)
    model.graph.node.insert(cst_idx,node)
    node=onnx.helper.make_node("Mul",inputs=[input_name,node_name+"_cst_out"],outputs=[node_name+"_out"],name=node_name)
    model.graph.node.insert(mul_idx,node)
    for item in model.graph.node:
        if item.name==ins_node_name:
            item.input[0]=node_name+"_out"
            break

def change_weight(node_name,new_value,model):
    for item in model.graph.initializer:
        if item.name==node_name:
            if hasattr(item,"raw_data"):
                item.raw_data=new_value.tobytes()
            else:
                assert False and "Not found raw_data"
        else:
            assert False and "Not found node_name {}".format(node_name)

if __name__=="__main__":
    input_data=np.load("input_data.npz")
    input_data=input_data["data"]
    input, conv_dict_input_data, attr,result_dict,shape_info=process("model.onnx",input_data=input_data)

    # print(result_dict.keys())

    model=onnx.load("model.onnx")

    for key in shape_info.keys():
        model.graph.output.extend([onnx.helper.make_tensor_value_info(key,onnx.TensorProto.FLOAT,shape_info[key])])

    weight_and_bias={}
    for item in model.graph.initializer:
        weight_and_bias[item.name]=item

    idx=-1
    conv_node={}
    # 拷贝一份model,深拷贝
    import copy
    model_copy=copy.deepcopy(model)
    for node in model_copy.graph.node:
        idx+=1
        if node.name=="ConvNd_1":
            continue
        if node.op_type=="Conv":
            node_name=node.name
            inputs=node.input
            ori_input=inputs[0]
            weight_dims=weight_and_bias[inputs[1]].dims
            raw=weight_and_bias[inputs[1]].raw_data
            weight=np.frombuffer(raw,dtype=np.float32)
            weight=weight.reshape(weight_dims)
            raw=weight_and_bias[inputs[2]].raw_data
            bias=np.frombuffer(raw,dtype=np.float32)
            bias=bias.reshape(-1)

            act_input=result_dict[ori_input]
            # act_input=act_input[0]
            # act_input=act_input.reshape(1,act_input.shape[0],act_input.shape[1],act_input.shape[2])

            dilations=node.attribute[1].ints
            group=node.attribute[2].i
            kernel_shape=node.attribute[3].ints
            pads=node.attribute[4].ints
            strides=node.attribute[5].ints

            pad_t,pad_l,pad_b,pad_r=pads[0],pads[1],pads[2],pads[3]
            stride_h,stride_w=strides[0],strides[1]
            dilation_h,dilation_w=dilations[0],dilations[1]

            N,C,H,W=act_input.shape
            C_out,C_in,kernel_h,kernel_w=weight.shape
            out_h=(H+pad_t+pad_b-(dilation_h*(kernel_h-1)+1))//stride_h+1
            out_w=(W+pad_l+pad_r-(dilation_w*(kernel_w-1)+1))//stride_w+1

            col=im2col_opt(act_input, kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, dilation_h, dilation_w, stride_h, stride_w)
            row=weight.reshape(C_out, -1).T
            ori_weight=row.T.reshape(C_out, C_in, kernel_h, kernel_w)
            ori_row=ori_weight.reshape(C_out, -1).T
            assert np.allclose(ori_row,row)
            assert np.allclose(ori_weight,weight)

            out=np.dot(col, row)+bias
            out=out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)

            col_abs=np.abs(col)
            col_max=np.max(col_abs,axis=0)
            row_abs=np.abs(row)
            row_max=np.max(row_abs,axis=1)
            scales=np.power(col_max,0.5)/np.power(row_max,0.5)

            ori_data=col2im(col, act_input.shape, kernel_h, kernel_w, stride_h, pad_t)
            assert np.allclose(ori_data,act_input)

            epsilon=1/(1<<31)
            print(epsilon)

            scales[scales==0]=1
            scales=np.ones_like(scales)
            assert not np.any(np.isinf(scales)) and not np.any(np.isnan(scales)) and not np.any(scales==0)
            scale_col=col/scales
            scale_row=row*scales.reshape(-1,1)

            scale_out=np.dot(col, row)+bias
            scale_out=scale_out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
            assert np.allclose(out,scale_out)

            scale_weight=scale_row.T.reshape(C_out, C_in, kernel_h, kernel_w)
            np.save("scales_weight.npy",scale_weight)
            scale_data=col2im(scale_col, act_input.shape, kernel_h, kernel_w, stride_h, pad_t)

            assert not np.allclose(scale_data,act_input)
            assert not np.allclose(scale_weight,weight)

            re_col=im2col(scale_data, kernel_h, kernel_w, pad_t, pad_l, pad_b, pad_r, dilation_h, dilation_w, stride_h, stride_w)
            re_row=scale_weight.reshape(C_out, -1).T
            assert np.allclose(re_col,scale_col)
            assert np.allclose(re_row,scale_row)

            new_out=np.dot(re_col, re_row)+bias
            new_out=new_out.reshape(N, out_h, out_w, C_out).transpose(0, 3, 1, 2)
            assert np.allclose(new_out,scale_out)
            assert np.allclose(new_out,out)

            scales_for_input=scale_data/act_input
            np.save("scales_for_input.npy",scales_for_input)
            assert np.allclose(scales_for_input*act_input,scale_data)

            make_mul_op("mul_"+str(idx),ori_input,act_input,0,idx+1,model,node_name)
            idx+=2

