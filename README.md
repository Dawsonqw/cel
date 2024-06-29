<!-- 简介 -->
# 项目简介
-- 目标：基于llvm、mlir实现一个端到端的编译器，实现对常用模型格式的解析到cpu、gpu的推理支持，包括不限于onnx、torch等常见格式，同时基于mlir实现对模型的优化加速和到目标代码的编译以及利用量化算法实现对模型的量化加速。

TODO:
- [x] 1. 抽象出中间IR用于表示模型
- [x] 2. 实现Onnx Parser，实现onnx模型到中间IR的转换
- [ ] 3. 实现对Onnx模型不做任何优化的推理，采用 cpu推理
- [ ] 4. 实现用CUDA对Onnx模型的推理
- [ ] 5. 实现mlir对计算图优化加速以及到LLVM IR的lowering，完成对onnx模型的优化加速
- [ ] 6. 实现对onnx模型的量化加速算法以及推理
- [ ] 7. 支持更多的模型格式，如torch等


<!-- 依赖安装 -->
# 依赖安装版本
```shell
# cmake version 3.28.3 
# glog version 0.7.0
# gtest version 1.14.0
# proto version 3.21.12

# protobuf 3.21.12 其它版本也可以，可能需要重新编译onnx.proto
apt-get install -y libprotobuf-dev protobuf-compiler
# onnx.proto
url=https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
# 产生onnx.pb.h onnx.pb.cc
protoc --cpp_out=./ onnx.proto
```