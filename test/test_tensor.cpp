#include <gtest/gtest.h>
#include "Parser/tensor.hpp"
#include <glog/logging.h>
#include <vector>

TEST(TensorTest, test_arma_major){
    google::InitGoogleLogging("test_tensor_major");
    google::SetStderrLogging(google::INFO);
    std::vector<float> data = {1,2,3,4};
    arma::Mat<float> mat(data.data(),2,2,false,true);
    LOG(INFO)<<"\n(0,0):"<<mat(0,0)<<"\n"<<
               "(0,1):"<<mat(0,1)<<"\n"<<
               "(1,0):"<<mat(1,0)<<"\n"<<
               "(1,1):"<<mat(1,1)<<"\n";

    if(mat(0,0)==1 && mat(0,1)==2 && mat(1,0)==3 && mat(1,1)==4){
        LOG(INFO)<<"arma为行主序";
    }else{
        LOG(INFO)<<"arma为列主序";
    }
}

TEST(TensorTest, test_arma_size){
    std::vector<float> data={1,2,3,4};
    arma::Cube<float> cube(data.data(),2,2,0,false,true);
    LOG(INFO)<<"\ncube.n_slices:"<<cube.n_slices<<"\n"<<
              "cube.n_rows:"<<cube.n_rows<<"\n"<<
              "cube.n_cols:"<<cube.n_cols<<"\n";
}

TEST(TensorTest, test_tensor_empty){
    // 初始化一个空的arma
    arma::Cube<float> cube;
    LOG(INFO)<<"is_emtpy: "<<cube.is_empty();
    LOG(INFO)<<"empty: "<<cube.empty();
    // 设置cuda的数据
    std::vector<float> data={1,2,3,4};
    cube = arma::Cube<float>(2,2,1);
    cube.slice(0) = arma::Mat<float>(data.data(),2,2,false,true);
    LOG(INFO)<<"is_emtpy: "<<cube.is_empty();
    LOG(INFO)<<"is_emtpy: "<<cube.empty();
    LOG(INFO)<<"\ncube.n_slices:"<<cube.n_slices<<"\n"<<
              "cube.n_rows:"<<cube.n_rows<<"\n"<<
              "cube.n_cols:"<<cube.n_cols<<"\n";
}