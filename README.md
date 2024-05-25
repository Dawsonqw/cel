<!-- 简介 -->
# 项目简介
-- 愿景：基于llvm、mlir实现一个端到端的编译器，实现基于python语法的前端语言到LLVM IR的编译，该语言主要做计算，期间利用mlir实现计算优化。

TODO:
- [ ] 1. 实现一个简单的前端语言，支持基本的计算
    - [ ] 1.1 实现词法分析器
    - [ ] 1.2 实现语法分析器
    - [ ] 1.3 实现解释器
- [ ] 2. 实现一个简单的编译器，将前端语言编译为LLVM IR
- [ ] 3. 实现一个简单的优化器，利用mlir实现计算优化
- [ ] 4. 实现一个简单的后端，将LLVM IR编译为目标代码
- [ ] 5. 实现一个简单的运行时，运行目标代码