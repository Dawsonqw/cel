#include <gtest/gtest.h>
#include <string>
#include <glog/logging.h>
#include "Parser/tensor.hpp"
#include <armadillo>
#include <vector>

TEST(Arma, matmul_flatten){
    google::InitGoogleLogging("test_arma");
    google::SetStderrLogging(google::INFO);

    // 定义matmul
    std::vector<float>data={1,2,3,4,5,6,7,8,9};
    std::vector<int>shape={3,3};

    // copy_aux_mem为true表示构造对象时数据独立，arma进行数据拷贝；为false表示数据共享，直接使用对应的内存；
    // strict为true表示表示在分配内存或进行操作时，会严格遵守给定的维度和大小。如果要求的尺寸不匹配或超出范围，将引发错误;为false时不会严格检查尺寸，可能导致错误
    arma::Mat<float> matmul(data.data(),shape[0],shape[1],false,false);
    LOG(INFO)<<"======= value =======";
    for(int i=0;i<shape[0];i++){
        for(int j=0;j<shape[1];j++){
            LOG(INFO)<<"matmul("<<i<<","<<j<<"):"<<matmul(i,j);
        }
    }
    // 转置
    LOG(INFO)<<"======= matmul.t() =======";
    arma::Mat<float> matmul_t = matmul.t();
    for(int i = 0; i < shape[1]; i++){
        for (int j = 0; j < shape[0]; j++){
            LOG(INFO)<<"matmul_t("<<i<<","<<j<<"):"<<matmul_t(i,j);
        }
    }

    // 按照rows进行flatten
    LOG(INFO)<<"======= matmul.flatten_rows() =======";
    arma::Mat<float> matmul_flatten_rows = matmul.t();
    matmul_flatten_rows.reshape(1,shape[0]*shape[1]);
    LOG(INFO)<<"rows: "<<matmul_flatten_rows.n_rows<<" cols: "<<matmul_flatten_rows.n_cols;
    for(int i=0;i<shape[0]*shape[1];i++){
        LOG(INFO)<<"matmul_flatten_rows("<<i<<"):"<<matmul_flatten_rows(i);
    }

    // 按照cols进行flatten
    LOG(INFO)<<"======= matmul.flatten_cols() =======";
    arma::Mat<float> matmul_flatten_cols = matmul;
    matmul_flatten_cols.reshape(1,shape[0]*shape[1]);
    LOG(INFO)<<"rows: "<<matmul_flatten_cols.n_rows<<" cols: "<<matmul_flatten_cols.n_cols;
    for(int i=0;i<shape[0]*shape[1];i++){
        LOG(INFO)<<"matmul_flatten_cols("<<i<<"):"<<matmul_flatten_cols(i);
    }
}

TEST(Arma, cube_flatten){


}