# awq算法实现
import numpy as np

# def get_act_scale(x):
#     return x.abs().view(-1, x.shape[-1]).mean(0)

def get_act_scale(input:np.array):
    return np.mean(np.abs(input),axis=0)

def awq_method(activation:np.array,weight:np.array,group_size:int):
    """
    awq算法实现
    :param activation: 激活值
    :param weight: 权重值
    :param group_size: 分组大小
    :return: scales
    """

    return