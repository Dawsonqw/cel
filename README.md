<!-- 简介 -->
# 项目简介
-- 目标：基于llvm、mlir实现一个端到端的编译器，实现对常用模型格式的解析到cpu、gpu的推理支持，包括不限于onnx、torch等常见格式，同时基于mlir实现对模型的优化加速和到目标代码的编译以及利用量化算法实现对模型的量化加速。

TODO:
- [x] 1. 抽象出中间IR用于表示模型
- [x] 2. 实现Onnx Parser，实现onnx模型到中间IR的转换
- [ ] 3. 实现对Onnx模型不做任何优化的推理，采用 cpu推理
    - [x] 3.1 实现对中间IR的拓扑排序和推理框架搭建
    - [x] 3.2 实现对resnet18的推理,完成数据的dump到二进制文件，可用numpy以np.frombuffer从该文件中读取数据
    - [ ] 3.3 逐层校验推理数据和onnxruntime的推理数据是否一致，以及shape是否一致
- [ ] 4. 实现用CUDA对Onnx模型的推理
- [ ] 5. 实现mlir对计算图优化加速以及到LLVM IR的lowering，完成对onnx模型的优化加速
- [ ] 6. 实现对onnx模型的量化加速算法以及推理
- [ ] 7. 支持更多的模型格式，如torch等

<!-- 新的TODO -->
<!-- 1.torch实现transformer模型，包括训练和推理 -->
<!-- 2.trition实现算子，优化推理 -->
<!-- 3.使用cuda实现算子，C++推理 -->
<!-- 4.使用pageattention flashattention优化推理代码 -->
<!-- 5.引入量化，awq、gptq实现以及kivi -->

new_todo:
- [ ] 1. torch实现transformer模型，包括训练和推理
- [ ] 2. 实现trition算子，优化推理
- [ ] 3. 实现cuda算子，C++推理
- [ ] 4. 实现pageattention flashattention优化推理代码
- [ ] 5. 引入量化，awq、gptq实现以及kivi


<!-- 依赖安装 -->
# 依赖安装版本
```shell
# cmake version 3.28.3 
# armadillo version 12.8.4
# glog version 0.7.0
# gtest version 1.14.0
# proto version 3.21.12
# llvm version 18.1.5

# protobuf 3.21.12 其它版本也可以，可能需要重新编译onnx.proto
apt-get install -y libprotobuf-dev protobuf-compiler
# onnx.proto
url=https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
# 产生onnx.pb.h onnx.pb.cc
protoc --cpp_out=./ onnx.proto

# llvm 编译命令
cmake -G Ninja ../llvm    \
            -DLLVM_ENABLE_PROJECTS="mlir"   \
            -DLLVM_ENABLE_RTTI=ON    \
            -DLLVM_TARGETS_TO_BUILD="host"   \
            -DCMAKE_BUILD_TYPE=Release   \
            -DLLVM_ENABLE_ASSERTIONS=ON  \
            -DLLVM_BUILD_EXAMPLES=ON \
            -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
            -DLLVM_LINK_LLVM_DYLIB=ON  \
            -DPython3_EXECUTABLE="python"

cmake --build . -j32&& ninja install


# triton
git clone https://github.com/triton-lang/triton.git;
cd triton/python;
pip install ninja cmake wheel; # build-time dependencies
pip install -e .

# torch
pip install troch
```