import torchvision
import torch

# 导出resnet18模型
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 保存模型为onnx
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True, input_names=['input'], output_names=['output'],opset_version=11)