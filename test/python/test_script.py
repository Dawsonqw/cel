import numpy as np
from tqdm import tqdm

def preudo_quantize(weight:np.array,group_size:int=0,zeor_point:bool=False):
        '''
        weight: np.array dimsize=2, shape=[n,m],权重数据
        group_size: int, 分组大小
        zeor_point: bool, 是否非对称量化
        '''
        org_w_shape = weight.shape
        if group_size > 0:
                assert org_w_shape[1] % group_size == 0
                weight = weight.reshape(-1, group_size)
        if zeor_point:
                # 获取每一行的最大值和最小值
                max_val = np.amax(weight, axis=1, keepdims=True)
                min_val = np.amin(weight, axis=1, keepdims=True)
                # 量化公式计算scale，zeop
                max_int = 2**8 - 1
                min_int = 0
                scales = (max_val - min_val).clip(min=1e-5) / max_int
                zeros = (-np.round(min_val / scales)).clip(min_int, max_int)
                # 量化并反量化
                weight = (np.clip(np.round(weight / scales) + zeros, min_int, max_int) - zeros) * scales
                zeros = zeros.reshape(-1)
                # assert zeros.shape[0] == org_w_shape[0]
        else:
                max_val = np.amax(np.abs(weight), axis=1, keepdims=True)
                max_val = np.clip(max_val, min=1e-5)
                max_int = 2 ** (8 - 1) - 1
                min_int = -(2 ** (8 - 1))
                scales = max_val / max_int
                zeros = None
                weight = np.clip(np.round(weight / scales), min_int, max_int) * scales
        
        scales = scales.reshape(-1)
        assert weight.shape[0] == scales.shape[0]
        weight = weight.reshape(org_w_shape)
        return weight,scales,zeros

def compute_scale_with_losss(input:np.array,weight:np.array,input_mean:np.array,weight_mean:np.array,
                                real_output:np.array,duo_scaling:bool,group_size:int,max_chunk_memory:int):
        '''
        input: np.array, 激活数据，shape=[m,n]
        weight: np.array, 权重数据，shape=[n,m]
        input_mean: np.array, 激活数据的平均值,每一列的平均值
        weight_mean: np.array, 权重数据的平均值,每一列的平均值
        real_output: np.array, 真实输出数据，fp32数据
        duo_scaling: bool, 是否
        '''
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")
        
        org_sd=weight
        
        x_mean=input_mean.reshape(-1)
        w_mean=weight_mean.reshape(-1)
        
        for ratio in tqdm(range(n_grid)):
                ratio = ratio / n_grid

                # 公式：s=sx^a · sw^(1-a)
                if duo_scaling:
                        scales=(x_mean**(ratio)/(w_mean**(1-ratio)+1e-4)).clip(min=1e-4)
                else:
                        scales=(x_mean**(ratio)/(w_mean+1e-4)).clip(min=1e-4)
                scales=scales/(np.sqrt(scales.max()*scales.min()))

                scales_view=scales.reshape(-1,1)

                scales[np.isinf(scales)]=1
                scales_view[np.isinf(scales)]=1
                
                # 获取放大后的权重
                scaled_weight=org_sd*scales_view
                # 获取根据放大后的并计算的反量化后的权重
                scaled_weight,_,_=preudo_quantize(scaled_weight,group_size,True)

                # W·s / s 后直接和w进行计算即可
                do_quant_weight=scaled_weight/scales_view

                def forward(input:np.array,do_quant_weight:np.array):
                        return (input@do_quant_weight)

                int_output=forward(input,do_quant_weight)

                loss=0.0
                
                f32_output=real_output.reshape(-1)
                int_output=int_output.reshape(-1)

                # 逐行计算loss
                num_elements=f32_output.size
                
                element_size_bytes=f32_output.itemsize
                
                chunk_size=int(max_chunk_memory//(element_size_bytes))
                chunk_size=min(chunk_size,num_elements)

                f32_chunks=np.split(f32_output,chunk_size)
                int_chunks=np.split(int_output,chunk_size)
                
                for f32_chunk,int_chunk in zip(f32_chunks,int_chunks):
                        chunk_loss=((f32_chunk-int_chunk).astype(np.float32)**2).sum()
                        loss+=chunk_loss
                        
                loss/=num_elements

                history.append(loss)
                if loss<best_error:
                        best_error=loss
                        best_ratio=ratio
                        best_scales=scales
                
        if best_ratio==-1:
                raise ValueError("No best ratio found")
        print("best_ratio:",best_ratio,"best_error:",best_error)
        return best_scales
                        
def search_best_scale(input:np.array,weight:np.array,real_output:np.array,group_size:int,max_chunk_memory:int):
        # 原始论文X scale计算列的平均值
        # Step1:将权重分组处理得到各组的归一化平均值
        ori_shape=weight.shape
        ori_weight=weight.copy()
        assert weight.size%group_size==0
        weight=weight.reshape(-1,group_size) # 按照group_size分组
        w_scale=np.abs(weight)/(np.abs(weight).max(axis=1,keepdims=True)+1e-6) # 归一化到[0,1]之间
        w_scale=w_scale.reshape(ori_shape)
        w_mean=w_scale.mean(1) # 计算每一个通道的归一化的平均值，这里计算每一列的平均值

        # Step2:计算激活数据逐chunk或者每一行的平均值
        input_flat=np.abs(input).reshape(-1,input.shape[-1])
        num_elements=input_flat.shape[0]
        num_channels=input_flat.shape[1]
        element_size_bytes=input_flat.itemsize
        
        chunk_size=int(max_chunk_memory//(element_size_bytes*num_channels))
        chunk_size=min(chunk_size,num_elements)
        
        x_sum=np.zeros(num_channels,dtype=np.float32)
        for i in range(0,num_elements,chunk_size):
            end=min(i+chunk_size,num_elements)
            chunk_sum=input_flat[i:end].astype(np.float32).sum(0)
            x_sum+=chunk_sum
            
        x_mean=x_sum/num_elements
        
        # assert np.allclose(x_mean,input_flat.mean(0))
        
        bset_scales=compute_scale_with_losss(input,ori_weight,x_mean,w_mean,real_output,True,group_size,max_chunk_memory)
        return bset_scales

def search_best_clip(input:np.array,weight:np.array,real_output:np.array,group_size:int,max_chunk_memory:int):
        '''
        input: np.array, 激活数据，shape=[m,n]
        weight: np.array, 权重数据，shape=[n,m]
        real_output: np.array, 真实输出数据，fp32数据
        group_size: int, 分组大小
        '''
        n_grid=20
        max_shrink=0.5
        n_sample=512
        
        ori_w_shape=weight.shape
        group_size=group_size if group_size>0 else ori_w_shape[1]
        
        input_feat=input
        input_feat=input_feat.reshape(-1,input_feat.shape[-1])
        input_feat=input_feat.reshape(1,input_feat.shape[0],-1,group_size)
        
        step_size=max(1,input_feat.shape[1]//n_sample)
        input_feat=input_feat[:,::step_size]
        
        w=weight.reshape(ori_w_shape[0],1,-1,group_size)
        
        oc_batch_size=256 if ori_w_shape[0]%256==0 else 64
        
        w_all=w
        best_max_val_all=[]
        
        # 通过网格搜索最大值
        print("search_best_clip")
        for i_b in tqdm(range(ori_w_shape[0]//oc_batch_size)):
                w=w_all[i_b*oc_batch_size:(i_b+1)*oc_batch_size]
                
                org_max_val=np.abs(w).max(axis=-1,keepdims=True)
                best_max_val=org_max_val.copy()
                min_errs=np.ones_like(org_max_val)*1e9
                org_out=(input_feat*w).sum(axis=-1)
                
                for i_s in range(int(max_shrink*n_grid)):
                        max_val=org_max_val*(1-i_s/n_grid)
                        min_val=-max_val
                        cur_w=np.clip(w,min_val,max_val)
                        q_w,_,_=preudo_quantize(cur_w,group_size,True)
                        cur_out=(input_feat*q_w).sum(axis=-1)
                        
                        err=((cur_out-org_out)**2).mean(axis=1).reshape(min_errs.shape)
                        cur_best_idx=err<min_errs
                        min_errs[cur_best_idx]=err[cur_best_idx]
                        best_max_val[cur_best_idx]=max_val[cur_best_idx]
                        
                best_max_val_all.append(best_max_val)
                
        best_max_val=np.concatenate(best_max_val_all,axis=0)
        
        return best_max_val.squeeze(1)

def apply_scale(input:np.array,scales:np.array):
        # 按照行进行除法
        ori_input_scales=input/scales
        return ori_input_scales

def apply_clip(weight:np.array,clip:np.array):
        ori_shape=weight.shape
        # 按照列进行clip，每一行的clip值不一样
        clip=clip.reshape(-1,1)
        assert clip.shape[0]==ori_shape[0] and "clip shape error"
        weight=np.clip(weight,-clip,clip)
        weight=weight.reshape(ori_shape)
        return weight
        
def quantize(input:np.array,weight:np.array,real_output:np.array,group_size:int,max_chunk_memory:int,do_clip:bool):
        scales=search_best_scale(input,weight,real_output,group_size,max_chunk_memory)
        ori_input=apply_scale(input,scales)
        scales=scales.reshape(-1,1)
        weight=weight*scales
        if do_clip:
                clip=search_best_clip(input,weight,real_output,group_size,max_chunk_memory)
                weight=apply_clip(weight,clip)
        return ori_input,weight

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
    padding_height_top,padding_width_left, padding_height_bottom,padding_width_right = paddings
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


def test_for_update_onnx_quantize(model_path:str,input_dict,outputs):
    import onnx 
    import onnxruntime as ort
    from onnx import numpy_helper
    model=onnx.load(model_path)
    onnx.checker.check_model(model)
    
    def construct_mul_node(mul_data:np.array,post_node:onnx.NodeProto,link_edge_name:str,node_name:str):
        '''
        构建一个乘法节点
        mul_data: 乘法节点的权重
        post_node: 后续节点
        link_edge_name: 前驱节点的输出名称
        node_name: 当前节点的名称   
        '''
        if len(mul_data.shape)==4:
            mean_data=np.mean(mul_data,axis=0)
            mul_data=mean_data
        initializers=[onnx.helper.make_tensor(name=link_edge_name+"_mul",data_type=onnx.TensorProto.FLOAT,\
                                                dims=mul_data.shape,vals=mul_data.flatten())]
        mul_node=onnx.helper.make_node("Mul",[link_edge_name,link_edge_name+"_mul"],[node_name+"_mul"],node_name+"_new_mul")
        input_dict=None
        for i in range(len(post_node.input)):
            if post_node.input[i]==link_edge_name:
                input_dict=(link_edge_name,node_name+"_mul")
                break
        return mul_node,initializers,input_dict

    def insert_mul_node(model,mul_node):
        input_link_name=mul_node.input[0]
        model_inputs=[input.name for input in model.graph.input]
        if input_link_name in model_inputs:
            model.graph.node.insert(0,mul_node)
            return
        for idx,node in enumerate(model.graph.node):
            if input_link_name in node.output:
                model.graph.node.insert(idx,mul_node)
                break

    def insert_initializers(model,init_node):
       model.graph.initializer.extend(init_node)

    def change_weight_data(model,weight_name,new_weight):
        for init in model.graph.initializer:
            if init.name==weight_name:
                if hasattr(init,"raw_data"):
                    init.raw_data=new_weight.tobytes()
                else:
                    init.CopyFrom(numpy_helper.from_array(new_weight))

    def activation_output(model,input_dict):
        for input in model.graph.input:
            input.type.tensor_type.shape.dim[0].dim_param="-1"
        for output in model.graph.output:
            output.type.tensor_type.shape.dim[0].dim_param="-1"

        node_atc={}
        for node in model.graph.node:
            if node.op_type=="Conv":
                link_node_name=node.input[0]
                node_atc[link_node_name]=node.name
        output_atc={}
        for node in model.graph.node:
            for output in node.output:
                if output in node_atc:
                    model.graph.output.extend([onnx.ValueInfoProto(name=output)])

        sess=ort.InferenceSession(model.SerializeToString())
        outputs=sess.get_outputs()
        outputs_list=[output.name for output in outputs]
        output=sess.run(outputs_list,input_dict)
        for i in range(len(outputs_list)):
            output_atc[outputs_list[i]]=output[i]

        return output_atc,model

    def get_init(model):
        init_dict={}
        for init in model.graph.initializer:
            init_dict[init.name]=numpy_helper.to_array(init)
        return init_dict
    

    import copy
    model_copy= copy.deepcopy(model)
    output_atc,update_model=activation_output(model_copy,input_dict)
    output_atc.update(input_dict)
    init_dict=get_init(model)
    input_tuples=[]
    all_nodes=[]
    for node in model.graph.node:
        if node.op_type=="Conv":
            all_nodes.append(node)

    skip_layer=[]
    for node in all_nodes:
        if node.op_type=="Conv":
            weight_name=node.input[1]
            if weight_name not in init_dict:
                raise ValueError("weight tensor not found")
            weight_tensor=init_dict[weight_name]
            if weight_tensor.shape[3]==1:
                skip_layer.append(node.name)
                continue
            dilations=node.attribute[0].ints
            input_tensor=output_atc[node.input[0]]
            conv_attributes = {attr.name: attr for attr in node.attribute}
        
            pads = conv_attributes.get('pads', None)
            strides = conv_attributes.get('strides', None)
            dilations = conv_attributes.get('dilations', None)
            group = conv_attributes.get('group', None)
            kernel_shape = conv_attributes.get('kernel_shape', None)
            if pads is not None:
                pads = pads.ints
            if strides is not None:
                strides = strides.ints
            if dilations is not None:
                dilations = dilations.ints
            if kernel_shape is not None:
                kernel_shape = kernel_shape.ints
            if group is not None:
                group = group.i
                
            input_feat=im2col_opt(input_tensor,kernel_shape[0],kernel_shape[1],
                                            pads[0],pads[1],pads[2],pads[3],dilations[0],dilations[1],strides[0],strides[1])
            weight_feat=weight_tensor.reshape(weight_tensor.shape[0],-1).T
            output=np.dot(input_feat,weight_feat)
            
            group_size=weight_feat.shape[1]
            max_chunk_memory=1024*1024*1024
            do_clip=False
            after_update_inputs,afer_update_weight=quantize(input_feat,weight_feat,output,group_size,max_chunk_memory,do_clip)
            # after_update_inputs,afer_update_weight=input_feat,weight_feat
            after_output=np.dot(after_update_inputs,afer_update_weight)
            if not np.allclose(output,after_output,atol=1e-5):
                raise ValueError("Test failed")

            ori_input= col2im_opt(after_update_inputs,input_tensor.shape,kernel_shape[0],kernel_shape[1],
                                            pads[0],pads[1],pads[2],pads[3],dilations[0],dilations[1],strides[0],strides[1])
            # ori_input=col2im(after_update_inputs,input_tensor.shape,kernel_shape,strides,pads,dilations)
            ori_weight=afer_update_weight.T.reshape(weight_tensor.shape)
            print("cur node: ",node.name)
            # if not np.allclose(ori_weight,weight_tensor,atol=1e-5):
            #     raise ValueError("Test weight failed",node.name)
            # if not np.allclose(ori_input,input_tensor,atol=1e-5):
            #     print("diff:",np.sum(np.abs(ori_input-input_tensor)))
            #     raise ValueError("Test oriinput failed",node.name)
                
            change_weight_data(model,weight_name,ori_weight)

            mul_datas=np.divide(ori_input,input_tensor,out=np.ones_like(ori_input),where=input_tensor!=0)

            mul_node,initializers,input_tupe=construct_mul_node(mul_datas,node,node.input[0],node.name)
            input_tuples.append((node.name,input_tupe[1]))
            insert_initializers(model,initializers)
            insert_mul_node(model,mul_node)
    update_names={}
    for input_tuple in input_tuples:
        update_names[input_tuple[0]]=input_tuple[1]
    for node in model.graph.node:
        if node.name in skip_layer:
            continue
        if node.op_type=="Conv":
            node.input[0]=update_names[node.name]
    model=onnx.shape_inference.infer_shapes(model)
    onnx.save(model,"quant.onnx")
    
    sess=ort.InferenceSession("quant.onnx")
    output=sess.run(outputs,input_dict)
    print(output[0].shape)
    
    ori_sess=ort.InferenceSession(model_path)
    ori_output=ori_sess.run(outputs,input_dict)
    print(ori_output[0].shape)
    
    cos=np.dot(output[0].flatten(),ori_output[0].flatten())/(np.linalg.norm(output[0])*np.linalg.norm(ori_output[0]))
    print("cos:",cos)

input_dict={
    "input":np.random.randn(1,3,224,224).astype(np.float32)
}
outputs=["output"]
test_for_update_onnx_quantize("resnet18.onnx",input_dict,outputs)