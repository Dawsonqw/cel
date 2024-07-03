import numpy as np

class QuantType:
    MIN_MAX = 0
    KL = 1

class DeQuant:
    def __init__(self, quant_data, scale, zero_point, bits, quant_type, quant_axis=None):
        self.quant_data = quant_data
        self.scale = scale
        self.zero_point = zero_point
        self.bits = bits
        self.quant_type = quant_type
        self.quant_axis = quant_axis

    def dequantize(self):
        if self.quant_type == QuantType.MIN_MAX:
            return self._dequantize_min_max()

        elif self.quant_type == QuantType.KL:
            return self._dequantize_kl()

    def _dequantize_min_max(self):
        if self.quant_axis is None:
            dequantized_data = (self.quant_data.astype(np.float32) - self.zero_point) * self.scale
        else:
            scale_shape = [1] * len(self.quant_data.shape)
            scale_shape[self.quant_axis] = self.scale.shape[0]
            zero_point_shape = [1] * len(self.quant_data.shape)
            zero_point_shape[self.quant_axis] = self.zero_point.shape[0]
            scale = self.scale.reshape(scale_shape)
            zero_point = self.zero_point.reshape(zero_point_shape)
            dequantized_data = (self.quant_data.astype(np.float32) - zero_point) * scale
        return dequantized_data

    def _dequantize_kl(self):
        pass