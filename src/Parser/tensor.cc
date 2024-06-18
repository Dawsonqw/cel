#include "Parser/tesnsor.hpp"

template <typename T>
cel::Tensor<T>::Tensor(T* raw_ptr, uint32_t size) {
  // CHECK_NE(raw_ptr, nullptr);
  this->raw_shapes_ = {size};
  this->m_data = arma::Cube<T>(raw_ptr, 1, size, 1, false, true);
}

template <typename T>
cel::Tensor<T>::Tensor(T* raw_ptr, uint32_t rows, uint32_t cols) {
  // CHECK_NE(raw_ptr, nullptr);
  this->m_data = arma::Cube<T>(raw_ptr, rows, cols, 1, false, true);
  if (rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  }
}

template <typename T>
cel::Tensor<T>::Tensor(T* raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols) {
  // CHECK_NE(raw_ptr, nullptr);
  this->m_data = arma::Cube<T>(raw_ptr, rows, cols, channels, false, true);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

template <typename T>
cel::Tensor<T>::Tensor(T* raw_ptr, const std::vector<uint32_t>& shapes) {
  // // CHECK_EQ(shapes.size(), 3);
  uint32_t channels = shapes.at(0);
  uint32_t rows = shapes.at(1);
  uint32_t cols = shapes.at(2);

  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }

  this->m_data = arma::Cube<T>(raw_ptr, rows, cols, channels, false, true);
}

template <typename T>
cel::Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  m_data = arma::Cube<T>(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

template <typename T>
cel::Tensor<T>::Tensor(uint32_t size) {
  m_data = arma::Cube<T>(1, size, 1);
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

template <typename T>
cel::Tensor<T>::Tensor(uint32_t rows, uint32_t cols) {
  m_data = arma::Cube<T>(rows, cols, 1);
  if (rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  }
}

template <typename T>
cel::Tensor<T>::Tensor(const std::vector<uint32_t>& shapes) {
  // CHECK(!shapes.empty() && shapes.size() <= 3);

  uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  uint32_t channels = shapes_.at(0);
  uint32_t rows = shapes_.at(1);
  uint32_t cols = shapes_.at(2);

  m_data = arma::Cube<T>(rows, cols, channels);
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

template <typename T>
uint32_t cel::Tensor<T>::rows() const {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.n_rows;
}

template <typename T>
uint32_t cel::Tensor<T>::cols() const {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.n_cols;
}

template <typename T>
uint32_t cel::Tensor<T>::channels() const {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.n_slices;
}

template <typename T>
size_t cel::Tensor<T>::size() const {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.size();
}

template <typename T>
void cel::Tensor<T>::set_data(const arma::Cube<T>& data) {
  // CHECK(data.n_rows == this->m_data.n_rows) << data.n_rows << " != " << this->m_data.n_rows;
  // CHECK(data.n_cols == this->m_data.n_cols) << data.n_cols << " != " << this->m_data.n_cols;
  // CHECK(data.n_slices == this->m_data.n_slices) << data.n_slices << " != " << this->m_data.n_slices;
  this->m_data = data;
}

template <typename T>
bool cel::Tensor<T>::empty() const {
  return this->m_data.empty();
}

template <typename T>
const T cel::Tensor<T>::index(uint32_t offset) const {
  // CHECK(offset < this->m_data.size()) << "Tensor index out of bound!";
  return this->m_data.at(offset);
}

template <typename T>
T& cel::Tensor<T>::index(uint32_t offset) {
  // CHECK(offset < this->m_data.size()) << "Tensor index out of bound!";
  return this->m_data.at(offset);
}

template <typename T>
std::vector<uint32_t> cel::Tensor<T>::shapes() const {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return {this->channels(), this->rows(), this->cols()};
}

template <typename T>
arma::Cube<T>& cel::Tensor<T>::data() {
  return this->m_data;
}

template <typename T>
const arma::Cube<T>& cel::Tensor<T>::data() const {
  return this->m_data;
}

template <typename T>
arma::Mat<T>& cel::Tensor<T>::slice(uint32_t channel) {
  // CHECK_LT(channel, this->channels());
  return this->m_data.slice(channel);
}

template <typename T>
const arma::Mat<T>& cel::Tensor<T>::slice(uint32_t channel) const {
  // CHECK_LT(channel, this->channels());
  return this->m_data.slice(channel);
}

template <typename T>
const T cel::Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  // CHECK_LT(row, this->rows());
  // CHECK_LT(col, this->cols());
  // CHECK_LT(channel, this->channels());
  return this->m_data.at(row, col, channel);
}

template <typename T>
T& cel::Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) {
  // CHECK_LT(row, this->rows());
  // CHECK_LT(col, this->cols());
  // CHECK_LT(channel, this->channels());
  return this->m_data.at(row, col, channel);
}

template <typename T>
void cel::Tensor<T>::Padding(const std::vector<uint32_t>& pads, T padding_value) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  // // CHECK_EQ(pads.size(), 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  arma::Cube<T> new_data(this->m_data.n_rows + pad_rows1 + pad_rows2,
                         this->m_data.n_cols + pad_cols1 + pad_cols2, this->m_data.n_slices);
  new_data.fill(padding_value);

  new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows2 - 1,
                   new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) = this->m_data;
  this->m_data = std::move(new_data);
  this->raw_shapes_ = std::vector<uint32_t>{this->channels(), this->rows(), this->cols()};
}

template <typename T>
void cel::Tensor<T>::Fill(T value) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  this->m_data.fill(value);
}

template <typename T>
void cel::Tensor<T>::Fill(const std::vector<T>& values, bool row_major) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  const uint32_t total_elems = this->m_data.size();
  // // CHECK_EQ(values.size(), total_elems);
  if (row_major) {
    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t planes = rows * cols;
    const uint32_t channels = this->channels();

    for (uint32_t i = 0; i < channels; ++i) {
      arma::Mat<T> channel_m_datat(const_cast<T*>(values.data()) + i * planes, this->cols(),
                                  this->rows(), false, true);
      this->m_data.slice(i) = channel_m_datat.t();
    }
  } else {
    std::copy(values.begin(), values.end(), this->m_data.memptr());
  }
}

template <typename T>
void cel::Tensor<T>::Flatten(bool row_major) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  const uint32_t size = this->m_data.size();
  this->Reshape({size}, row_major);
}

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

#ifdef _MSC_VER
  std::uniform_int_distribution<int32_t> dist(min, max);
  uint8_t max_value = std::numeric_limits<uint8_t>::max();
  for (uint32_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt) % max_value;
  }
#else
  std::uniform_int_distribution<std::uint8_t> dist(min, max);
  for (size_t i = 0; i < this->size(); ++i) {
    this->index(i) = dist(mt);
  }
#endif
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

template <typename T>
void cel::Tensor<T>::Ones() {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  this->Fill(T{1});
}

template <typename T>
void cel::Tensor<T>::Transform(const std::function<T(T)>& filter) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  this->m_data.transform(filter);
}

template <typename T>
const std::vector<uint32_t>& cel::Tensor<T>::raw_shapes() const {
  // CHECK(!this->raw_shapes_.empty());
  // CHECK_LE(this->raw_shapes_.size(), 3);
  // CHECK_GE(this->raw_shapes_.size(), 1);
  return this->raw_shapes_;
}

template <typename T>
void cel::Tensor<T>::Reshape(const std::vector<int32_t>& shapes, bool row_major) {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  // CHECK(!shapes.empty());
  const size_t origin_size = this->size();
  const size_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), size_t(1), std::multiplies<size_t>());
  // CHECK(shapes.size() <= 3);
  // CHECK(current_size == origin_size);
  if (!row_major) {
    if (shapes.size() == 3) {
      this->m_data.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
      this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
      this->m_data.reshape(shapes.at(0), shapes.at(1), 1);
      this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
    } else {
      this->m_data.reshape(1, shapes.at(0), 1);
      this->raw_shapes_ = {shapes.at(0)};
    }
  } else {
    if (shapes.size() == 3) {
      this->Review({shapes.at(0), shapes.at(1), shapes.at(2)});
      this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
      this->Review({1, shapes.at(0), shapes.at(1)});
      this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
    } else {
      this->Review({1, 1, shapes.at(0)});
      this->raw_shapes_ = {shapes.at(0)};
    }
  }
}

template <typename T>
T* cel::Tensor<T>::raw_ptr() {
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.memptr();
}

template <typename T>
const T* cel::Tensor<T>::raw_ptr() const {
  return this->m_data.memptr();
}

template <typename T>
const T* cel::Tensor<T>::raw_ptr(size_t offset)const {
  const size_t size = this->size();
  // CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  // CHECK_LT(offset, size);
  return this->m_data.memptr() + offset;
}

template <typename T>
std::vector<T> cel::Tensor<T>::values(bool row_major) {
  // // CHECK_EQ(this->m_data.empty(), false);
  std::vector<T> values(this->m_data.size());

  if (!row_major) {
    std::copy(this->m_data.mem, this->m_data.mem + this->m_data.size(), values.begin());
  } else {
    uint32_t index = 0;
    for (uint32_t c = 0; c < this->m_data.n_slices; ++c) {
      const arma::Mat<T>& channel = this->m_data.slice(c).t();
      std::copy(channel.begin(), channel.end(), values.begin() + index);
      index += channel.size();
    }
    // // CHECK_EQ(index, values.size());
  }
  return values;
}