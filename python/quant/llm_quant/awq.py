import numpy as np

input=np.random.rand(2,3).astype(np.float32)
weight=np.random.rand(3,4).astype(np.float32)

def preudo_quantize(weight:np.array,group_size:int=0,zeor_point:bool=False):
        org_w_shape = weight.shape
        if group_size > 0:
                weight = weight.reshape(-1, group_size)
        
        if zeor_point:
                max_val = np.amax(weight, axis=1, keepdims=True)
                min_val = np.amin(weight, axis=1, keepdims=True)
                max_int = 2**8 - 1
                min_int = 0
                scales = (max_val - min_val).clip(min=1e-5) / max_int
                zeros = (-np.round(min_val / scales)).clip(min_int, max_int)
                weight = (np.clip(np.round(weight / scales) + zeros, min_int, max_int) - zeros) * scales
                zeros = zeros.reshape(org_w_shape[0], -1)
        else:
                max_val = np.amax(np.abs(weight), axis=1, keepdims=True)
                max_val = np.clip(max_val, min=1e-5)
                max_int = 2 ** (8 - 1) - 1
                min_int = -(2 ** (8 - 1))
                scales = max_val / max_int
                zeros = None
                weight = np.clip(np.round(weight / scales), min_int, max_int) * scales
        
        scales = scales.reshape(org_w_shape[0], -1)
        weight = weight.reshape(org_w_shape)
        return weight,scales,zeros

def pseudo_quantize_tensor(weight:np.array,zero_point:bool):
        ori_shape=weight.shape
        
        if zero_point:
                max_val=np.abs(weight).max(axis=-1,keepdims=True)
                min_val=-max_val
                max_int=2**8-1
                min_int=0
                scales=(max_val-min_val).clip(min=1e-5)/max_int
                zeros=(-np.round(min_val/scales)).clip(min_int,max_int)
                weight=(np.clip(np.round(weight/scales)+zeros,min_int,max_int)-zeros)*scales
                zeros=zeros.reshape(ori_shape[0],-1)
        else:
                max_val=np.abs(weight).max(axis=-1,keepdims=True)
                max_val=np.clip(max_val,min=1e-5)
                max_int=2**(8-1)-1
                min_int=-(2**(8-1))
                scales=max_val/max_int
                zeros=None
                weight=np.clip(np.round(weight/scales),min_int,max_int)*scales
        
def compute_scale_with_losss(input:np.array,weight:np.array,input_mean:np.array,weight_mean:np.array,
                                real_output:np.array,duo_scaling:bool,group_size:int,max_chunk_memory:int):
        n_grid = 20
        history = []
        best_ratio = -1
        best_scales = None
        best_error = float("inf")
        
        org_sd=weight
        
        x_mean=input_mean.reshape(-1)
        w_mean=weight_mean.reshape(-1)

        for ratio in range(n_grid):
                ratio = ratio / n_grid

                if duo_scaling:
                        scales=(x_mean.pow(ratio)/(w_mean.pow(1-ratio)+1e-4)).clamp(min=1e-4)
                else:
                        scales=(x_mean.pow(ratio)/(w_mean+1e-4)).clamp(min=1e-4)
                scales=scales/(scales.max()*scales.min()).sqrt()
                scales_view=scales.reshape(-1)

                scales[np.isinf(scales)]=1
                scales_view[np.isinf(scales)]=1

                scaled_weight=org_sd*scales_view
                preudo_scale_weight,_,_=preudo_scale_weight(scaled_weight,group_size,True)
                do_quant_weight=preudo_scale_weight

                def forward(input:np.array,do_quant_weight:np.array):
                        # todo:计算该过程的推理结果
                        return None
                int_output=forward(input,do_quant_weight)

                loss=0.0
                
                f32_output=real_output.reshape(-1)
                int_output=int_output.reshape(-1)

                num_elements=f32_output.size
                element_size_bytes=f32_output.itemsize*2
                
                chunk_size=int(max_chunk_memory//(element_size_bytes*2))
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

        return best_scales
                        
def search_best_scale(input:np.array,weight:np.array,real_output:np.array,group_size:int,max_chunk_memory:int):
        ori_shape=weight.shape
        ori_weight=weight.copy()
        weight=weight.reshape(-1,group_size)
        w_scale=np.abs(weight)/(np.abs(weight).max(axis=1,keepdims=True)+1e-6)
        w_scale=w_scale.reshape(ori_shape)
        w_mean=w_scale.mean(0)

        input_flat=np.abs(input).reshape(-1,input.shape[-1])
        num_elements=input_flat.shape[0]
        num_channels=input_flat.shape[1]
        element_size_bytes=input_flat.itemsize*2
        
        chunk_size=int(max_chunk_memory//(element_size_bytes*num_channels))
        chunk_size=min(chunk_size,num_elements)
        
        x_sum=np.zeros(num_channels,dtype=np.float32)
        for i in range(0,num_elements,chunk_size):
            end=min(i+chunk_size,num_elements)
            chunk_sum=input_flat[i:end].astype(np.float32).sum(0)
            x_sum+=chunk_sum
            
        x_mean=x_sum/num_elements

        bset_scales=compute_scale_with_losss(input,ori_weight,w_mean,x_mean,real_output,True,group_size,max_chunk_memory)
        return bset_scales

def search_best_clip(input:np.array,weight:np.array,real_output:np.array,group_size:int,max_chunk_memory:int):
        n_grid=20
        max_shrink=0.5
        n_sample=512
        
        ori_w_shape=weight.shape
        group_size=group_size if group_size>0 else ori_w_shape[1]
        
        input_feat=input_feat.reshape(-1,input_feat.shape[-1])
        input_feat=input_feat.reshape(1,input_feat.shape[0],-1,group_size)
        
        step_size=max(1,input_feat.shape[1]//n_sample)
        input_feat=input_feat[:,::step_size]
        
        w=weight.reshape(ori_w_shape[0],1,-1,group_size)
        
        oc_batch_size=256 if ori_w_shape[0]%256==0 else 64
        
        w_all=w
        best_max_val_all=[]
        
        for i_b in range(ori_w_shape[0]//oc_batch_size):
                w=w_all[i_b*oc_batch_size:(i_b+1)*oc_batch_size]
                
                org_max_val=np.abs(w).max(axis=-1,keepdims=True)
                best_max_val=org_max_val.copy()
                min_errs=np.ones_like(org_max_val)*1e9
                input_feat=input_feat.to(w.device)
                org_out=(input_feat*w).sum(axis=-1)
                
                for i_s in range(int(max_shrink*n_grid)):
                        max_val=org_max_val*(1-i_s/n_grid)
                        min_val=-max_val
                        cur_w=np.clip(w,min_val,max_val)
                        q_w=pseudo_quantize_tensor(cur_w)[0]
                        cur_out=(input_feat*q_w).sum(axis=-1)
                        
                        err=((cur_out-org_out)**2).mean(axis=1).reshape(min_errs.shape)
                        cur_best_idx=err<min_errs
                        min_errs[cur_best_idx]=err[cur_best_idx]
                        best_max_val[cur_best_idx]=max_val[cur_best_idx]
                        
                best_max_val_all.append(best_max_val)
                
        best_max_val=np.concatenate(best_max_val_all,axis=0)
        
        return best_max_val.squeeze(1)

def apply_scale(input:np.array,scales:np.array):
        mul_scales=np.array()
        return mul_scales

def apply_clip(weight:np.array,clip:np.array):
        ori_shape=weight.shape
        weight=weight.reshape(-1,clip.shape[-1])
        weight=np.clip(weight,-clip,clip)
        weight=weight.reshape(ori_shape)
        return weight
        
def quantize(input:np.array,weight:np.array,real_output:np.array,group_size:int,max_chunk_memory:int,do_clip:bool):
        scales=search_best_scale(input,weight,real_output,group_size,max_chunk_memory)
        mul_valus=apply_scale(input,scales)
        
        if do_clip:
                clip=search_best_clip(input,weight,real_output,group_size,max_chunk_memory)
                weight=apply_clip(weight,clip)
        
        return mul_valus,weight