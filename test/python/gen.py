# import torchvision
# import torch

# # 导出resnet18模型
# model = torchvision.models.resnet18(pretrained=True)
# model.eval()

# # 保存模型为onnx
# dummy_input = torch.randn(1, 3, 224, 224)
# torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True, input_names=['input'], output_names=['output'],opset_version=11)

import onnx
from onnx import shape_inference
model=onnx.load("resnet18.onnx")
# shape infer,onnx.shape_inference.infer_shapes(model)
infer_model=shape_inference.infer_shapes(model)
onnx.save(infer_model,"resnet18_infer.onnx")