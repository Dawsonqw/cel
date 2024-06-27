#include "Parser/tensor.hpp"

template <>
void cel::Tensor<float>::RandN(float mean, float var) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  std::random_device rd;
  std::mt19937 mt(rd());

  std::normal_distribution<float> dist(mean, var);
  for (size_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt);
  }
}

template <>
void cel::Tensor<int32_t>::RandU(int32_t min, int32_t max) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  std::random_device rd;
  std::mt19937 mt(rd());

  std::uniform_int_distribution<int32_t> dist(min, max);
  for (size_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt);
  }
}

template <>
void cel::Tensor<std::uint8_t>::RandU(std::uint8_t min, std::uint8_t max) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  std::random_device rd;
  std::mt19937 mt(rd());

  std::uniform_int_distribution<std::uint8_t> dist(min, max);
  for (size_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt);
  }
}

template <>
void cel::Tensor<float>::RandU(float min, float max) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  // CHECK(max >= min);
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(min, max);
  for (size_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt);
  }
}