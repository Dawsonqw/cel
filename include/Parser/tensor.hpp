#ifndef BASE_TENSOR_H
#define BASE_TENSOR_H
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>
#include <glog/logging.h>
#include <iostream>

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
            explicit Tensor(const Tensor<T>& tensor);
            ~Tensor();

            int32_t rows() const;
            int32_t cols() const;
            int32_t channels() const;
            size_t size() const;
            size_t plane_size() const;
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
            void set_data(const arma::Cube<T>& data);
            void set_data(int32_t channels, int32_t rows, int32_t cols,T value);
            void set_data(int32_t rows, int32_t cols,T value);
            void set_data(int32_t index,T value);
            void Ones();
            void RandN(T mean = 0, T var = 1);
            void RandU(T min = 0, T max = 1);
            void Reshape(const std::vector<int32_t>& shapes, bool row_major = false);
            void Review(const std::vector<uint32_t>& shapes);
            void Permute(const std::vector<int32_t>& dims);
            void Transpose();
            void Flatten(bool row_major = false);
            void Flatten(int64_t axis);
            void Transform(const std::function<T(T)>& filter);
            const T* raw_ptr() const;
            const T* raw_ptr(size_t offset) const;
            T* raw_ptr();
            T* raw_ptr(size_t offset);
            T* matrix_raw_ptr(uint32_t index);
            const T* matrix_raw_ptr(uint32_t index) const;
            void dump(const std::string& path,bool row_major=false,bool append=true) const;
            void load(const std::string& path,const std::vector<int32_t>& shapes,size_t offset,bool row_major=false);

        private:
            std::vector<int32_t> m_shape;
            arma::Cube<T> m_data;
    };
}

// armadilloe是列主序，因此size被设置在cols上
template <typename T>
cel::Tensor<T>::Tensor(T* raw_ptr, int32_t size) {
  CHECK_NE(raw_ptr, nullptr);
  this->m_shape = std::vector<int32_t>{size};
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

  if(shapes.size()==1){
    m_data = arma::Cube<T>(1, shapes.at(0), 1);
  }else if(shapes.size()==2){
    m_data = arma::Cube<T>(shapes.at(0), shapes.at(1), 1);
  }else{
    m_data = arma::Cube<T>(shapes.at(1), shapes.at(2), shapes.at(0));
  }
  this->m_shape = shapes;
}

template <typename T> inline cel::Tensor<T>::Tensor(const Tensor<T> &tensor) {
  this->m_data = tensor.m_data;
  this->m_shape = tensor.m_shape;
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

template <typename T> inline size_t cel::Tensor<T>::plane_size() const { 
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  return this->rows() * this->cols();
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

template <typename T> inline void cel::Tensor<T>::Flatten(int64_t axis) {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  CHECK(axis >= 0 && axis < 3);
  if (axis == 0) {
    this->Reshape({this->size()}, false);
  } else if (axis == 1) {
    this->Reshape({1,this->rows()*this->cols()}, false);
  } else {
    this->Reshape({1,this->rows() ,this->cols()}, false);
  }
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
      this->m_shape=std::vector<int32_t>{shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
      this->m_data.reshape(shapes.at(0), shapes.at(1), 1);
      this->m_shape=std::vector<int32_t>{shapes.at(0), shapes.at(1)};
    } else {
      this->m_data.reshape(1, shapes.at(0), 1);
      this->m_shape=std::vector<int32_t>{shapes.at(0)};
    }
  } else {
    if (shapes.size() == 3) {
      this->Review({shapes.at(0), shapes.at(1), shapes.at(2)});
      this->m_shape=std::vector<int32_t>{shapes.at(0), shapes.at(1), shapes.at(2)};
    } else if (shapes.size() == 2) {
      this->Review({1, shapes.at(0), shapes.at(1)});
      this->m_shape=std::vector<int32_t>{shapes.at(0), shapes.at(1)};
    } else {
      this->Review({1, 1, shapes.at(0)});
      this->m_shape=std::vector<int32_t>{shapes.at(0)};
    }
  }
}


template <typename T>
void cel::Tensor<T>::Review(const std::vector<uint32_t>& shapes) {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  CHECK_EQ(shapes.size(), 3);
  const uint32_t target_ch = shapes.at(0);
  const uint32_t target_rows = shapes.at(1);
  const uint32_t target_cols = shapes.at(2);

  CHECK_EQ(this->m_data.size(), target_ch * target_cols * target_rows);
  arma::Cube<T> new_data(target_rows, target_cols, target_ch);
  const uint32_t plane_size = target_rows * target_cols;
#pragma omp parallel for
  for (uint32_t channel = 0; channel < this->m_data.n_slices; ++channel) {
    const uint32_t plane_start = channel * m_data.n_rows * m_data.n_cols;
    for (uint32_t src_col = 0; src_col < this->m_data.n_cols; ++src_col) {
      const T* col_ptr = this->m_data.slice_colptr(channel, src_col);
      for (uint32_t src_row = 0; src_row < this->m_data.n_rows; ++src_row) {
        const uint32_t pos_idx = plane_start + src_row * m_data.n_cols + src_col;
        const uint32_t dst_ch = pos_idx / plane_size;
        const uint32_t dst_ch_offset = pos_idx % plane_size;
        const uint32_t dst_row = dst_ch_offset / target_cols;
        const uint32_t dst_col = dst_ch_offset % target_cols;
        new_data.at(dst_row, dst_col, dst_ch) = *(col_ptr + src_row);
      }
    }
  }
  this->m_data = std::move(new_data);
}

template <typename T> inline void cel::Tensor<T>::Permute(const std::vector<int32_t> &dims) {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  CHECK_EQ(dims.size(), 3);
  std::vector<int32_t> new_shape = {this->m_shape.at(dims.at(0)), this->m_shape.at(dims.at(1)), this->m_shape.at(dims.at(2))};

  if (new_shape == this->m_shape) {
      return;
  }

  Tensor<T> temp(new_shape);

  for (int32_t i = 0; i < this->m_shape[0]; ++i) {
      for (int32_t j = 0; j < this->m_shape[1]; ++j) {
          for (int32_t k = 0; k < this->m_shape[2]; ++k) {
              int32_t old_index = i * this->m_shape[1] * this->m_shape[2] + j * this->m_shape[2] + k;
              int32_t new_i = i * this->m_shape[1] * this->m_shape[2] / (this->m_shape[dims[0]] * this->m_shape[dims[1]] * this->m_shape[dims[2]]);
              int32_t new_j = (i * this->m_shape[1] * this->m_shape[2] % (this->m_shape[dims[0]] * this->m_shape[dims[1]] * this->m_shape[dims[2]])) / (this->m_shape[dims[1]] * this->m_shape[dims[2]]);
              int32_t new_k = (i * this->m_shape[1] * this->m_shape[2] % (this->m_shape[dims[1]] * this->m_shape[dims[2]])) / this->m_shape[dims[2]];
              int32_t new_index = new_i * new_shape[1] * new_shape[2] + new_j * new_shape[2] + new_k;
              temp.m_data[new_index] = this->m_data[old_index];
          }
      }
  }
  this->m_shape = new_shape;
  this->m_data = std::move(temp.m_data);
}

template <typename T> inline void cel::Tensor<T>::Transpose() {
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  arma::Cube<T> transposed_cube(this->m_data.n_cols,this->m_data.n_rows,this->m_data.n_slices);
  for(size_t index=0;index<this->m_data.n_slices;index++){
    transposed_cube.slice(index)=this->m_data.slice(index).t();
  }
  this->m_data=std::move(transposed_cube);
  if(this->m_shape.size()==2){
    this->m_shape=std::vector<int32_t>{this->m_shape.at(1),this->m_shape.at(0)};
  }
  else if(this->m_shape.size()==3){
    this->m_shape=std::vector<int32_t>{this->m_shape.at(1),this->m_shape.at(0),this->m_shape.at(2)};
  }
  else{
    LOG(ERROR)<<"shape size must be less than 3";
  }
}

template <typename T> inline void cel::Tensor<T>::set_size(const std::vector<int32_t> &shapes) {
  if(shapes.size()==1){
    this->m_data.set_size(1,shapes.at(0),1);
    this->m_shape=std::vector<int32_t>{shapes.at(0)};
  }
  else if(shapes.size()==2){
    this->m_data.set_size(shapes.at(0),shapes.at(1),1);
    this->m_shape=std::vector<int32_t>{shapes.at(0),shapes.at(1)};
  }
  else if(shapes.size()==3){
    this->m_data.set_size(shapes.at(1),shapes.at(2),shapes.at(0));
    this->m_shape=std::vector<int32_t>{shapes.at(0),shapes.at(1),shapes.at(2)};
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
  this->m_data(rows, cols, channels) = value;
}

template <typename T> inline void cel::Tensor<T>::set_data(int32_t rows, int32_t cols, T value) {
  CHECK_GE(rows, 0);
  CHECK_GE(cols, 0);
  CHECK_LE(rows, this->rows());
  CHECK_LE(cols, this->cols());
  this->m_data(rows, cols, 0) = value;
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

template <typename T> inline T *cel::Tensor<T>::raw_ptr(size_t offset) { 
  const size_t size = this->size();
  CHECK(!this->m_data.empty()) << "The data area of the tensor is empty.";
  CHECK_LT(offset, size);
  return this->m_data.memptr() + offset;
 }

template<typename T>
inline T * cel::Tensor<T>::matrix_raw_ptr(uint32_t index)
{
  CHECK_LT(index, this->channels());
  size_t offset = index * this->plane_size();
  CHECK_LE(offset, this->size());
  T* mem_ptr = this->raw_ptr(offset);
  return mem_ptr;
}

template <typename T> inline const T *cel::Tensor<T>::matrix_raw_ptr(uint32_t index) const {
  CHECK_LT(index, this->channels());
  size_t offset = index * this->plane_size();
  CHECK_LE(offset, this->size());
  T* mem_ptr = this->raw_ptr(offset);
  return mem_ptr;
}

template <typename T> inline void cel::Tensor<T>::dump(const std::string &path,bool raw_major,bool append) const {
  std::ofstream file(path, std::ios::binary);
  CHECK(file.is_open()) << "Failed to open file: " << path;
  const size_t size = this->size();
  if (!raw_major) {
    file.write(reinterpret_cast<const char*>(this->m_data.memptr()), size * sizeof(T));
  } else {
    for (int32_t c = 0; c < this->m_data.n_slices; ++c) {
      const arma::Mat<T>& channel = this->m_data.slice(c).t();
      if(append){
        file.seekp(0,std::ios::end);
      }
      else{
        file.seekp(0,std::ios::beg);
      }
      file.write(reinterpret_cast<const char*>(channel.memptr()), channel.size() * sizeof(T));
    }
  }
  file.close();
}

template <typename T>
inline void cel::Tensor<T>::load(const std::string &path, const std::vector<int32_t> &shapes,size_t offset,
                                 bool row_major) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.is_open()) << "Failed to open file: " << path;
  this->set_size(shapes);
  file.seekg(offset * sizeof(T), std::ios::beg);
  const size_t size = this->size();
  if (!row_major) {
    file.read(reinterpret_cast<char*>(const_cast<T*>(this->m_data.memptr())), size * sizeof(T));
  } else {
    for (int32_t c = 0; c < this->m_data.n_slices; ++c) {
      const arma::Mat<T>& channel = this->m_data.slice(c).t();
      file.read(reinterpret_cast<char*>(const_cast<T*>(channel.memptr())), channel.size() * sizeof(T));
    }
  }
  file.close();
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