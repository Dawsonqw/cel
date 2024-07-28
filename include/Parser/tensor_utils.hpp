#ifndef BASE_TENSOR_UTIL_H
#define BSSE_TENSOR_UTIL_H
#include "tensor.hpp"
#include <functional>
#include <armadillo>

namespace cel {
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
    LOG_IF(FATAL,tensor_lhs->rows()!=tensor_rhs->rows()||tensor_lhs->cols()!=tensor_rhs->cols())<<"tensor_lhs shape must be equal to tensor_rhs shape";
    arma::Mat<T> mat_lhs(tensor_lhs->data().memptr(),tensor_lhs->rows(),tensor_lhs->cols(),false,true);
    arma::Mat<T> mat_rhs(tensor_rhs->data().memptr(),tensor_rhs->rows(),tensor_rhs->cols(),false,true);
    arma::Mat<T> mat_result=mat_lhs+mat_rhs;
    std::vector<int32_t> shape={mat_result.n_rows,mat_result.n_cols};
    return std::make_shared<Tensor<T>>(mat_result.memptr(),shape);
  }

}
#endif 