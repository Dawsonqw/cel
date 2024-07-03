# 实现基础的量化算法

import numpy as np

def symmetric_quantize(x, bits):
    # 找到数据的最大绝对值
    max_val = np.max(np.abs(x))
    # 计算缩放因子
    scale = max_val / (2**(bits - 1) - 1)
    # 量化
    x_quantized = np.round(x / scale)
    return x_quantized, scale

def symmetric_dequantize(x_quantized, scale):
    # 反量化
    x_dequantized = x_quantized * scale
    return x_dequantized

# 示例
data = np.array([1.2, -3.4, 0.7, -1.5])
bits = 8

quantized_data, scale = symmetric_quantize(data, bits)
dequantized_data = symmetric_dequantize(quantized_data, scale)

print("量化数据:", quantized_data)
print("反量化数据:", dequantized_data)