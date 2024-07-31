#ifndef BASE_TENSOR_UTIL_H
#define BSSE_TENSOR_UTIL_H
#include "tensor.hpp"
#include <functional>
#include <armadillo>

namespace cel {
template <typename T>
  std::tuple<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> TensorBroadcast(
      const std::shared_ptr<Tensor<T>>& tensor_lhs, const std::shared_ptr<Tensor<T>>& tensor_rhs) {
    CHECK(tensor_lhs != nullptr && tensor_rhs != nullptr);
    auto lhs_shapes = tensor_lhs->shapes();
    auto rhs_shapes = tensor_rhs->shapes();

    int32_t max_dim = std::max(lhs_shapes.size(), rhs_shapes.size());
    std::vector<int32_t> broadcasted_lhs(max_dim, 1);
    std::vector<int32_t> broadcasted_rhs(max_dim, 1);

    for (int32_t i = 0; i < max_dim; ++i) {
        broadcasted_lhs[max_dim - i - 1] = (i < lhs_shapes.size()) ? lhs_shapes[lhs_shapes.size() - i - 1] : 1;
        broadcasted_rhs[max_dim - i - 1] = (i < rhs_shapes.size()) ? rhs_shapes[rhs_shapes.size() - i - 1] : 1;
    }

    for (int32_t i = 0; i < max_dim; ++i) {
        if (broadcasted_lhs[i] != broadcasted_rhs[i] && broadcasted_lhs[i] != 1 && broadcasted_rhs[i] != 1) {
            throw std::invalid_argument("Cannot broadcast tensors with incompatible shapes.");
        }
    }

    std::vector<int32_t> final_shape(max_dim);
    for (int32_t i = 0; i < max_dim; ++i) {
        final_shape[i] = std::max(broadcasted_lhs[i], broadcasted_rhs[i]);
    }

    auto tensor_lhs_broadcasted = std::make_shared<Tensor<T>>(final_shape);
    auto tensor_rhs_broadcasted = std::make_shared<Tensor<T>>(final_shape);

    auto lhs_data = tensor_lhs->data();
    auto rhs_data = tensor_rhs->data();
    auto lhs_broadcasted_data = tensor_lhs_broadcasted->data();
    auto rhs_broadcasted_data = tensor_rhs_broadcasted->data();

    // 填充lhs广播后的数据
    for (arma::uword i = 0; i < lhs_broadcasted_data.n_elem; ++i) {
        arma::uword idx = i;
        arma::uword lhs_idx = 0;
        arma::uword lhs_stride = lhs_data.n_elem / tensor_lhs->shapes().back();
        for (int32_t dim = max_dim - 1; dim >= 0; --dim) {
            int32_t lhs_dim_size = (dim < lhs_shapes.size()) ? lhs_shapes[dim] : 1;
            lhs_idx += (idx % final_shape[dim]) % lhs_dim_size * lhs_stride;
            lhs_stride /= lhs_dim_size;
            idx /= final_shape[dim];
        }
        lhs_broadcasted_data[i] = lhs_data[lhs_idx];
    }

    // 填充rhs广播后的数据
    for (arma::uword i = 0; i < rhs_broadcasted_data.n_elem; ++i) {
        arma::uword idx = i;
        arma::uword rhs_idx = 0;
        arma::uword rhs_stride = rhs_data.n_elem / tensor_rhs->shapes().back();
        for (int32_t dim = max_dim - 1; dim >= 0; --dim) {
            int32_t rhs_dim_size = (dim < rhs_shapes.size()) ? rhs_shapes[dim] : 1;
            rhs_idx += (idx % final_shape[dim]) % rhs_dim_size * rhs_stride;
            rhs_stride /= rhs_dim_size;
            idx /= final_shape[dim];
        }
        rhs_broadcasted_data[i] = rhs_data[rhs_idx];
    }

    return std::make_tuple(tensor_lhs_broadcasted, tensor_rhs_broadcasted);
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> create_tensor(const std::vector<T>& data, const std::vector<int32_t>& shape){
    auto get_size = [](const std::vector<int32_t>& shape){
      int32_t size = 1;
      for(auto s:shape){
        size *= s;
      }
      return size;
    };
    LOG_IF(FATAL,get_size(shape)!=data.size())<<"data size must be equal to shape size";
    return std::make_shared<Tensor<T>>(data.data(),shape);
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> matmul(const std::shared_ptr<Tensor<T>>& tensor_lhs,const std::shared_ptr<Tensor<T>>& tensor_rhs){
    LOG_IF(FATAL,tensor_lhs->cols()!=tensor_rhs->rows())<<"tensor_lhs cols must be equal to tensor_rhs rows";
    arma::Mat<T> mat_lhs(tensor_lhs->data().memptr(),tensor_lhs->rows(),tensor_lhs->cols(),false,true);
    arma::Mat<T> mat_rhs(tensor_rhs->data().memptr(),tensor_rhs->rows(),tensor_rhs->cols(),false,true);
    arma::Mat<T> mat_result=mat_lhs*mat_rhs;
    std::vector<int32_t> shape={mat_result.n_rows,mat_result.n_cols};
    return std::make_shared<Tensor<T>>(mat_result.memptr(),shape);
  }


  template<typename T>
  std::shared_ptr<Tensor<T>> add(const std::shared_ptr<Tensor<T>>& tensor_lhs,const std::shared_ptr<Tensor<T>>& tensor_rhs){
    CHECK(tensor_lhs != nullptr && tensor_rhs != nullptr );
    std::shared_ptr<Tensor<T>> output_tensor = std::make_shared<Tensor<T>>(tensor_lhs->shapes());
    if (tensor_lhs->shapes() == tensor_rhs->shapes()) {
      CHECK(tensor_lhs->shapes() == output_tensor->shapes());
      output_tensor->set_data(tensor_lhs->data() + tensor_rhs->data());
    } else {
      CHECK(tensor_lhs->channels() == tensor_rhs->channels()) << "Tensors shape are not adapting";
      const auto& [input_tensor_lhs, input_tensor_rhs] = TensorBroadcast(tensor_lhs, tensor_rhs);
      CHECK(output_tensor->shapes() == input_tensor_lhs->shapes() &&
            output_tensor->shapes() == input_tensor_rhs->shapes());
      output_tensor->set_data(input_tensor_lhs->data() + input_tensor_rhs->data());
    }
    if(output_tensor->shapes()!=tensor_lhs->raw_shapes()){
      output_tensor->Reshape(tensor_lhs->raw_shapes());
    }
    return output_tensor;
  }

  template<typename T>
  std::shared_ptr<Tensor<T>> mul(const std::shared_ptr<Tensor<T>>& tensor_lhs,const std::shared_ptr<Tensor<T>>& tensor_rhs){
    CHECK(tensor_lhs != nullptr && tensor_rhs != nullptr );
    std::shared_ptr<Tensor<T>> output_tensor = std::make_shared<Tensor<T>>(tensor_lhs->shapes());
    if (tensor_lhs->shapes() == tensor_rhs->shapes()) {
      CHECK(tensor_lhs->shapes() == output_tensor->shapes());
      output_tensor->set_data(tensor_lhs->data() % tensor_rhs->data());
    } else {
      CHECK(tensor_lhs->channels() == tensor_rhs->channels()) << "Tensors shape are not adapting";
      const auto& [input_tensor_lhs, input_tensor_rhs] = TensorBroadcast(tensor_lhs, tensor_rhs);
      CHECK(output_tensor->shapes() == input_tensor_lhs->shapes() &&
            output_tensor->shapes() == input_tensor_rhs->shapes());
      output_tensor->set_data(input_tensor_lhs->data() % input_tensor_rhs->data());
    }
    return output_tensor;
  }

}
#endif 