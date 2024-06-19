#include <gtest/gtest.h>
#include "Parser/tensor.hpp"

#include <vector>
#include <iostream>
using namespace cel;

TEST(TensorTest, test_tensor_chw){
    std::vector<float>vec_data(10, 1.0);
    float* data = vec_data.data();
    std::vector<int32_t> shapes = {1,2, 5};
    cel::Tensor<float> tensor(data,shapes);
    ASSERT_EQ(tensor.rows(), 2);
    ASSERT_EQ(tensor.cols(), 5);
    ASSERT_EQ(tensor.channels(), 1);
}

TEST(TensorTest, test_tensor_size){
    std::vector<float>vec_data(10, 1.0);
    float* data = vec_data.data();
    std::vector<int32_t> shapes = {2, 5, 1};
    cel::Tensor<float> tensor(data,shapes);
    ASSERT_EQ(tensor.size(), 10);
}

TEST(TensorTest, test_tensor_empty){
    std::vector<float>vec_data(10, 1.0);
    float* data = vec_data.data();
    std::vector<int32_t> shapes = {2, 5, 1};
    cel::Tensor<float> tensor(data,shapes);
    ASSERT_EQ(tensor.empty(), false);
    vec_data.clear();
    data=vec_data.data();
    cel::Tensor<float> tensor2(data,0);
    ASSERT_EQ(tensor2.empty(), true);
}