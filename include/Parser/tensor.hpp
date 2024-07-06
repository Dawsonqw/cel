#ifndef BASE_TENSOR_H
#define BASE_TENSOR_H
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>
#include <glog/logging.h>

namespace cel{
    template<typename T=double>
    class Tensor{
        public:
            explicit Tensor() = default;
            explicit Tensor(T* raw_ptr, int32_t size);
            explicit Tensor(T* raw_ptr, int32_t rows, int32_t cols);
            explicit Tensor(T* raw_ptr, int32_t channels, int32_t rows, int32_t cols);
            explicit Tensor(T* raw_ptr, const std::vector<int32_t>& shapes);
            explicit Tensor(int32_t channels, int32_t rows, int32_t cols);
            explicit Tensor(int32_t size);
            explicit Tensor(int32_t rows, int32_t cols);
            explicit Tensor(const std::vector<int32_t>& shapes);
            ~Tensor();

            int32_t rows() const;
            int32_t cols() const;
            int32_t channels() const;
            size_t size() const;
            void set_data(const arma::Cube<T>& data);
            bool empty() const;
            T& index(int32_t offset);
            const T index(int32_t offset) const;
            std::vector<int32_t> shapes() const;
            const std::vector<int32_t>& raw_shapes() const;
            arma::Cube<T>& data();
            const arma::Cube<T>& data() const;
            arma::Mat<T>& slice(int32_t channel);
            const arma::Mat<T>& slice(int32_t channel) const;
            const T at(int32_t channel, int32_t row, int32_t col) const;
            T& at(int32_t channel, int32_t row, int32_t col);
            void Padding(const std::vector<int32_t>& pads, T padding_value);
            void Fill(T value);
            void Fill(const std::vector<T>& values, bool row_major = true);
            std::vector<T> values(bool row_major = true);
            void set_size(const std::vector<int32_t>& shapes);
            void set_data(int32_t channels, int32_t rows, int32_t cols,T value);
            void set_data(int32_t rows, int32_t cols,T value);
            void set_data(int32_t index,T value);
            void Ones();
            void RandN(T mean = 0, T var = 1);
            void RandU(T min = 0, T max = 1);
            void Reshape(const std::vector<int32_t>& shapes, bool row_major = false);
            void Flatten(bool row_major = false);
            void Transform(const std::function<T(T)>& filter);
            const T* raw_ptr() const;
            const T* raw_ptr(size_t offset) const;
            T* raw_ptr();
        private:
            std::vector<int32_t> m_shape;
            arma::Cube<T> m_data;
    };
}

// armadilloe是列主序，因此size被设置在cols上
template <typename T>
cel::Tensor<T>::Tensor(T* raw_ptr, int32_t size) {
  CHECK_NE(raw_ptr, nullptr);
  this->m_shape = {size};
  this->m_data = arma::Cube<T>(raw_ptr, 1, size, 1, false, true);
}

template <typename T>
cel::Tensor<T>::Tensor(T* raw_ptr, int32_t rows, int32_t cols) {
  CHECK_NE(raw_ptr, nullptr);
  this->m_data = arma::Cube<T>(raw_ptr, rows, cols, 1, false, true);
  this->m_shape = std::vector<int32_t>{rows, cols};
}

template <typename T>
cel::Tensor<T>::Tensor(T* raw_ptr, int32_t channels, int32_t rows, int32_t cols) {
  CHECK_NE(raw_ptr, nullptr);
  this->m_data = arma::Cube<T>(raw_ptr, rows, cols, channels, false, true);
  this->m_shape = std::vector<int32_t>{channels, rows, cols};
}

template <typename T>
cel::Tensor<T>::Tensor(T* raw_ptr, const std::vector<int32_t>& shapes) {
  CHECK_NE(raw_ptr, nullptr);
  CHECK_GT(shapes.size(),0);
  this->m_shape = shapes;
  if(shapes.size()==1){
    this->m_data=arma::Cube<T>(raw_ptr,1,shapes.at(0),1,false,true);
  }else if(shapes.size()==2){
    this->m_data=arma::Cube<T>(raw_ptr,shapes.at(0),shapes.at(1),1,false,true);
  }else if(shapes.size()==3){
    this->m_data=arma::Cube<T>(raw_ptr,shapes.at(1),shapes.at(2),shapes.at(0),false,true);
  }
  else{
    LOG(ERROR)<<"shape size must be less than 3";
  }
}

template <typename T>
cel::Tensor<T>::Tensor(int32_t channels, int32_t rows, int32_t cols) {
  m_data = arma::Cube<T>(rows, cols, channels);
  this->m_shape = std::vector<int32_t>{channels, rows, cols};
}

template <typename T>
cel::Tensor<T>::Tensor(int32_t size) {
  m_data = arma::Cube<T>(1, size, 1);
  this->m_shape = std::vector<int32_t>{size};
}

template <typename T>
cel::Tensor<T>::Tensor(int32_t rows, int32_t cols) {
  m_data = arma::Cube<T>(rows, cols, 1);
  this->m_shape = std::vector<int32_t>{rows, cols};
}

template <typename T>
cel::Tensor<T>::Tensor(const std::vector<int32_t>& shapes) {
  CHECK(!shapes.empty() && shapes.size() <= 3);

  int32_t channels = shapes.at(0);
  int32_t rows = shapes.at(1);
  int32_t cols = shapes.at(2);

  m_data = arma::Cube<T>(rows, cols, channels);
  this->m_shape = std::vector<int32_t>{channels, rows, cols};
}

template <typename T>
cel::Tensor<T>::~Tensor(){
}

template <typename T>
int32_t cel::Tensor<T>::rows() const {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.n_rows;
}

template <typename T>
int32_t cel::Tensor<T>::cols() const {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.n_cols;
}

template <typename T>
int32_t cel::Tensor<T>::channels() const {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.n_slices;
}

template <typename T>
size_t cel::Tensor<T>::size() const {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.size();
}

template <typename T>
void cel::Tensor<T>::set_data(const arma::Cube<T>& data) {
  CHECK(data.n_rows == this->m_data.n_rows) << data.n_rows << " != " << this->m_data.n_rows;
  CHECK(data.n_cols == this->m_data.n_cols) << data.n_cols << " != " << this->m_data.n_cols;
  CHECK(data.n_slices == this->m_data.n_slices) << data.n_slices << " != " << this->m_data.n_slices;
  this->m_data = data;
}

template <typename T>
bool cel::Tensor<T>::empty() const {
  return this->m_data.empty();
}

template <typename T>
const T cel::Tensor<T>::index(int32_t offset) const {
  CHECK(offset < this->m_data.size()) << "Tensor index out of bound!";
  return this->m_data.at(offset);
}

template <typename T>
T& cel::Tensor<T>::index(int32_t offset) {
  CHECK(offset < this->m_data.size()) << "Tensor index out of bound!";
  return this->m_data.at(offset);
}

template <typename T>
std::vector<int32_t> cel::Tensor<T>::shapes() const {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
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
arma::Mat<T>& cel::Tensor<T>::slice(int32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->m_data.slice(channel);
}

template <typename T>
const arma::Mat<T>& cel::Tensor<T>::slice(int32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->m_data.slice(channel);
}

template <typename T>
const T cel::Tensor<T>::at(int32_t channel, int32_t row, int32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->m_data.at(row, col, channel);
}

template <typename T>
T& cel::Tensor<T>::at(int32_t channel, int32_t row, int32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->m_data.at(row, col, channel);
}

template <typename T>
void cel::Tensor<T>::Padding(const std::vector<int32_t>& pads, T padding_value) {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  CHECK_EQ(pads.size(), 4);
  int32_t pad_rows1 = pads.at(0);  // up
  int32_t pad_rows2 = pads.at(1);  // bottom
  int32_t pad_cols1 = pads.at(2);  // left
  int32_t pad_cols2 = pads.at(3);  // right

  arma::Cube<T> new_data(this->m_data.n_rows + pad_rows1 + pad_rows2,
                         this->m_data.n_cols + pad_cols1 + pad_cols2, this->m_data.n_slices);
  new_data.fill(padding_value);

  new_data.subcube(pad_rows1, pad_cols1, 0, new_data.n_rows - pad_rows2 - 1,
                   new_data.n_cols - pad_cols2 - 1, new_data.n_slices - 1) = this->m_data;
  this->m_data = std::move(new_data);
  this->m_shape = std::vector<int32_t>{this->channels(), this->rows(), this->cols()};
}

template <typename T>
void cel::Tensor<T>::Fill(T value) {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  this->m_data.fill(value);
}

template <typename T>
void cel::Tensor<T>::Fill(const std::vector<T>& values, bool row_major) {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  const int32_t total_elems = this->m_data.size();
  CHECK_EQ(values.size(), total_elems);
  if (row_major) {
    const int32_t rows = this->rows();
    const int32_t cols = this->cols();
    const int32_t planes = rows * cols;
    const int32_t channels = this->channels();

    for (int32_t i = 0; i < channels; ++i) {
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
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  const int32_t size = this->m_data.size();
  this->Reshape({size}, row_major);
}

template <typename T>
void cel::Tensor<T>::RandN(T mean, T var)
{
  LOG(ERROR) << "Not implemented yet!";
}

template <typename T>
void cel::Tensor<T>::RandU(T min, T max)
{
  LOG(ERROR) << "Not implemented yet!";
}

template <typename T>
void cel::Tensor<T>::Ones() {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  this->Fill(T{1});
}

template <typename T>
void cel::Tensor<T>::Transform(const std::function<T(T)> &filter)
{
    CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
    this->m_data.transform(filter);
}

template <typename T>
const std::vector<int32_t>& cel::Tensor<T>::raw_shapes() const {
  CHECK(!this->m_shape.empty());
  return this->m_shape;
}

template <typename T>
void cel::Tensor<T>::Reshape(const std::vector<int32_t>& shapes, bool row_major) {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  CHECK(!shapes.empty());
  const size_t origin_size = this->size();
  const size_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), size_t(1), std::multiplies<size_t>());
  CHECK(shapes.size() <= 3);
  CHECK(current_size == origin_size);
  if (!row_major) {
    if (shapes.size() == 3) {
      this->m_data.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
      this->m_shape = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
      this->m_data.reshape(shapes.at(0), shapes.at(1), 1);
      this->m_shape = {shapes.at(0), shapes.at(1)};
    } else {
      this->m_data.reshape(1, shapes.at(0), 1);
      this->m_shape = {shapes.at(0)};
    }
  } else {
    if (shapes.size() == 3) {
      this->Review({shapes.at(0), shapes.at(1), shapes.at(2)});
      this->m_shape = {shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
      this->Review({1, shapes.at(0), shapes.at(1)});
      this->m_shape = {shapes.at(0), shapes.at(1)};
    } else {
      this->Review({1, 1, shapes.at(0)});
      this->m_shape = {shapes.at(0)};
    }
  }
}

template <typename T> inline void cel::Tensor<T>::set_size(const std::vector<int32_t> &shapes) {
  if(shapes.size()==1){
    this->m_data.set_size(1,shapes.at(0),1);
    this->m_shape = {shapes.at(0)};
  }
  else if(shapes.size()==2){
    this->m_data.set_size(shapes.at(0),shapes.at(1),1);
    this->m_shape = {shapes.at(0),shapes.at(1)};
  }
  else if(shapes.size()==3){
    this->m_data.set_size(shapes.at(1),shapes.at(2),shapes.at(0));
    this->m_shape = {shapes.at(0),shapes.at(1),shapes.at(2)};
  }
  else{
    LOG(ERROR)<<"shape size must be less than 3";
  }
}

template <typename T>
inline void cel::Tensor<T>::set_data(int32_t channels, int32_t rows, int32_t cols,T value) {
  CHECK_GE(channels, 0);
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  CHECK_LE(channels, this->channels());
  CHECK_LE(rows, this->rows());
  CHECK_LE(cols, this->cols());
  this->m_data(channels, rows, cols) = value;
}

template <typename T> inline void cel::Tensor<T>::set_data(int32_t rows, int32_t cols, T value) {
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  CHECK_LE(rows, this->rows());
  CHECK_LE(cols, this->cols());
  this->m_data(0, rows, cols) = value;
}

template <typename T> inline void cel::Tensor<T>::set_data(int32_t index, T value) {
  CHECK_GE(index, 0);
  CHECK_LT(index, this->size());
  this->m_data.at(index) = value;
}

template <typename T>
T* cel::Tensor<T>::raw_ptr() {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->m_data.memptr();
}

template <typename T>
const T* cel::Tensor<T>::raw_ptr() const {
  return this->m_data.memptr();
}

template <typename T>
const T* cel::Tensor<T>::raw_ptr(size_t offset)const {
  const size_t size = this->size();
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  CHECK_LT(offset, size);
  return this->m_data.memptr() + offset;
}

template <typename T>
std::vector<T> cel::Tensor<T>::values(bool row_major) {
  CHECK_EQ(this->m_data.empty(), false);
  std::vector<T> values(this->m_data.size());

  if (!row_major) {
    std::copy(this->m_data.mem, this->m_data.mem + this->m_data.size(), values.begin());
  } else {
    int32_t index = 0;
    for (int32_t c = 0; c < this->m_data.n_slices; ++c) {
      const arma::Mat<T>& channel = this->m_data.slice(c).t();
      std::copy(channel.begin(), channel.end(), values.begin() + index);
      index += channel.size();
    }
    CHECK_EQ(index, values.size());
  }
  return values;
}


#endif