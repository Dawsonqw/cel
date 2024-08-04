import numpy as np
from quant.awq import quantize
from quant.utils import im2col,col2im,im2col_opt,col2im_opt


def test_quantize():
    input=np.random.rand(128,64).astype(np.float32)
    weight=np.random.rand(64,64).astype(np.float32)
    
    real_output=np.random.rand(8,2).astype(np.float32)
    group_size=64
    max_chunk_memory=1024*1024*1024
    
    for_construct_mul_op,afer_update_weight=quantize(input,weight,real_output,group_size,max_chunk_memory,True)

def test_conv_quantize():
    input=np.random.rand(1,64,56,56).astype(np.float32)
    weight=np.random.rand(64,64,3,3).astype(np.float32)
    dilation=np.array([1,1])
    paddings=np.array([1,1,1,1])
    strides=np.array([1,1])
    kernel_shape=weight.shape[2:]
    
    # input_feat=im2col(input,kernel_shape,strides,paddings,dilation)
    input_feat=im2col_opt(input,3,3,1,1,1,1,1,1,1,1)
    weight_feat=weight.reshape(weight.shape[0],-1).T
    print("input shape:",input_feat.shape)
    print("weight shape:",weight_feat.shape)
    output=np.dot(input_feat,weight_feat)

    group_size=weight_feat.shape[1]
    max_chunk_memory=1024*1024*1024
    do_clip=False
    after_update_inputs,afer_update_weight=quantize(input_feat,weight_feat,output,group_size,max_chunk_memory,do_clip)

    after_output=np.dot(after_update_inputs,afer_update_weight)
    print("diff:",np.sum(np.abs(output-after_output)))
    print(np.allclose(output,after_output,atol=1e-4))

    if not np.allclose(output,after_output,atol=1e-4):
        print("Test failed")
        exit(-1)
    ori_input= col2im(after_update_inputs,input.shape,kernel_shape,strides,paddings,dilation)
    # ori_input=col2im_opt(after_update_inputs,input.shape,3,3,1,1,1,1,1,1,1)
    ori_weight=afer_update_weight.T.reshape(weight.shape)
    
    mul_datas=ori_input/input
    print(mul_datas.shape)

def test_for_update_onnx_quantize():
    import onnx 
    import onnxruntime as ort
    from onnx import numpy_helper
    model=onnx.load("resnet18.onnx")
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
    
    input_dict={
        "input":np.random.randn(1,3,224,224).astype(np.float32)
    }
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
            after_update_inputs,afer_update_weight,scales=quantize(input_feat,weight_feat,output,group_size,max_chunk_memory,do_clip)
            # after_update_inputs,afer_update_weight=input_feat,weight_feat
            after_output=np.dot(after_update_inputs,afer_update_weight)
            if not np.allclose(output,after_output,atol=1e-5):
                raise ValueError("Test failed")
            else:
                print("quant layer of success".format(node.name))

            ori_input= col2im_opt(after_update_inputs,input_tensor.shape,kernel_shape[0],kernel_shape[1],
                                            pads[0],pads[1],pads[2],pads[3],dilations[0],dilations[1],strides[0],strides[1])
            # ori_input=col2im(after_update_inputs,input_tensor.shape,kernel_shape,strides,pads,dilations)
            ori_weight=afer_update_weight.T.reshape(weight_tensor.shape)
            print("cur node: ",node.name)
            # if not np.allclose(ori_weight,weight_tensor,atol=1e-4):
            #     raise ValueError("Test weight failed",node.name)
            # if not np.allclose(ori_input,input_tensor,atol=1e-4):
            #     print("diff:",np.sum(np.abs(ori_input-input_tensor)))
            #     raise ValueError("Test oriinput failed",node.name)
                
            change_weight_data(model,weight_name,ori_weight)


            # input_tensor+1e-15
            input_tensor=input_tensor+1e-12
            mul_datas=np.divide(ori_input,input_tensor,out=np.zeros_like(ori_input),where=input_tensor!=0)

            scaled_input=input_tensor*mul_datas
            # if np.allclose(scaled_input,ori_input,atol=1e-4):
            #     print("sum_diff: ",np.sum(np.abs(scaled_input-ori_input)))
            #     print("max_diff: ",np.abs(scaled_input-ori_input))
                
            #     raise ValueError("mul_datas failed",node.name)

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
    onnx.save(model,"resnet18_quant.onnx")
    
    # 推理onnxruntime，比较前后的模型输出是否一致
    input_dict={
        "input":np.random.randn(1,3,224,224).astype(np.float32)
    }
    
    sess=ort.InferenceSession("resnet18_quant.onnx")
    outputs=["output"]
    output=sess.run(outputs,input_dict)
    print(output[0].shape)
    
    ori_sess=ort.InferenceSession("resnet18.onnx")
    ori_output=ori_sess.run(outputs,input_dict)
    print(ori_output[0].shape)
    
    cos=np.dot(output[0].flatten(),ori_output[0].flatten())/(np.linalg.norm(output[0])*np.linalg.norm(ori_output[0]))
    print("cos:",cos)

def test_smoothquant_for_update_onnx_quantize():
    import onnx 
    import onnxruntime as ort
    from onnx import numpy_helper
    model=onnx.load("resnet18.onnx")
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
    
    input_dict={
        "input":np.random.randn(1,3,224,224).astype(np.float32)
    }
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
            
            col_abs=np.abs(input_feat)
            col_max=np.max(col_abs,axis=0)
            row_abs=np.abs(weight_feat)
            row_max=np.max(row_abs,axis=1)
            scales=np.power(col_max,0.5)/np.power(row_max,0.5)
        
            scales[scales==0]=1
            scales=np.ones_like(scales)
            assert not np.any(np.isinf(scales)) and not np.any(np.isnan(scales)) and not np.any(scales==0)
            scale_col=input_feat/scales
            scale_row=weight_feat*scales.reshape(-1,1)
            
            after_update_inputs,afer_update_weight=scale_col,scale_row
            after_output=np.dot(after_update_inputs,afer_update_weight)
            if not np.allclose(output,after_output,atol=1e-5):
                raise ValueError("Test failed")
            else:
                print("quant layer of success".format(node.name))

            ori_input= col2im_opt(after_update_inputs,input_tensor.shape,kernel_shape[0],kernel_shape[1],
                                            pads[0],pads[1],pads[2],pads[3],dilations[0],dilations[1],strides[0],strides[1])
            # ori_input=col2im(after_update_inputs,input_tensor.shape,kernel_shape,strides,pads,dilations)
            ori_weight=afer_update_weight.T.reshape(weight_tensor.shape)
            print("cur node: ",node.name)
            # if not np.allclose(ori_weight,weight_tensor,atol=1e-4):
            #     raise ValueError("Test weight failed",node.name)
            # if not np.allclose(ori_input,input_tensor,atol=1e-4):
            #     print("diff:",np.sum(np.abs(ori_input-input_tensor)))
            #     raise ValueError("Test oriinput failed",node.name)
                
            change_weight_data(model,weight_name,ori_weight)

            # input_tensor+1e-15
            input_tensor=input_tensor+1e-12
            mul_datas=np.divide(ori_input,input_tensor,out=np.zeros_like(ori_input),where=input_tensor!=0)

            scaled_input=input_tensor*mul_datas
            # if np.allclose(scaled_input,ori_input,atol=1e-4):
            #     print("sum_diff: ",np.sum(np.abs(scaled_input-ori_input)))
            #     print("max_diff: ",np.abs(scaled_input-ori_input))
                
            #     raise ValueError("mul_datas failed",node.name)

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
    onnx.save(model,"resnet18_quant.onnx")
    
    # 推理onnxruntime，比较前后的模型输出是否一致
    input_dict={
        "input":np.random.randn(1,3,224,224).astype(np.float32)
    }
    
    sess=ort.InferenceSession("resnet18_quant.onnx")
    outputs=["output"]
    output=sess.run(outputs,input_dict)
    print(output[0].shape)
    
    ori_sess=ort.InferenceSession("resnet18.onnx")
    ori_output=ori_sess.run(outputs,input_dict)
    print(ori_output[0].shape)
    
    cos=np.dot(output[0].flatten(),ori_output[0].flatten())/(np.linalg.norm(output[0])*np.linalg.norm(ori_output[0]))
    print("cos:",cos)
# test_quantize()
# test_conv_quantize()

# test_for_update_onnx_quantize()
    
# test_quantize()
# test_conv_quantize()

test_smoothquant_for_update_onnx_quantize()