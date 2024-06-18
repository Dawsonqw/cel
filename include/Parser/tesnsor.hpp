#ifndef BASE_TENSOR_H
#define BASE_TENSOR_H
#include <armadillo>
#include <memory>
#include <numeric>
#include <vector>

namespace cel{
    template<typename T=float>
    class Tensor{
        public:
            explicit Tensor(T* raw_ptr, uint32_t size);
            explicit Tensor(T* raw_ptr, uint32_t rows, uint32_t cols);
            explicit Tensor(T* raw_ptr, uint32_t channels, uint32_t rows, uint32_t cols);
            explicit Tensor(T* raw_ptr, const std::vector<uint32_t>& shapes);
            explicit Tensor() = default;
            explicit Tensor(uint32_t channels, uint32_t rows, uint32_t cols);
            explicit Tensor(uint32_t size);
            explicit Tensor(uint32_t rows, uint32_t cols);
            explicit Tensor(const std::vector<uint32_t>& shapes);
            uint32_t rows() const;
            uint32_t cols() const;
            uint32_t channels() const;
            size_t size() const;
            void set_data(const arma::Cube<T>& data);
            bool empty() const;
            T& index(uint32_t offset);
            const T index(uint32_t offset) const;
            std::vector<uint32_t> shapes() const;
            const std::vector<uint32_t>& raw_shapes() const;
            arma::Cube<T>& data();
            const arma::Cube<T>& data() const;
            arma::Mat<T>& slice(uint32_t channel);
            const arma::Mat<T>& slice(uint32_t channel) const;
            const T at(uint32_t channel, uint32_t row, uint32_t col) const;
            T& at(uint32_t channel, uint32_t row, uint32_t col);
            void Padding(const std::vector<uint32_t>& pads, T padding_value);
            void Fill(T value);
            void Fill(const std::vector<T>& values, bool row_major = true);
            std::vector<T> values(bool row_major = true);
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

#endif