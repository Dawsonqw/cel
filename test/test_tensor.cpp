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

TEST(TensorTest, test_tensor_data){
    std::vector<float> data={1,2,3,4,5,6};
    cel::Tensor<float> tensor(data.data(),{1,2,3});
    // 1 3 5
    // 2 4 6
    LOG(INFO)<<"rows: "<<tensor.rows();
    LOG(INFO)<<"cols: "<<tensor.cols();
    LOG(INFO)<<"slices: "<<tensor.channels();
    for(int i=0;i<6;i++){
        LOG(INFO)<<"tensor["<<i<<"]:"<<tensor.index(i);
    }
    for(int i=0;i<tensor.rows();i++){
        for(int j=0;j<tensor.cols();j++){
                LOG(INFO)<<"tensor["<<i<<","<<j<<","<<"]:"<<tensor.at(tensor.channels()-1,i,j);
        }
    }
    tensor=cel::Tensor<float>(data.data(),{1,3,2});
    // 1 4
    // 2 5
    // 3 6
    LOG(INFO)<<"rows: "<<tensor.rows();
    LOG(INFO)<<"cols: "<<tensor.cols();
    LOG(INFO)<<"slices: "<<tensor.channels();
    for(int i=0;i<6;i++){
        LOG(INFO)<<"tensor["<<i<<"]:"<<tensor.index(i);
    }
    for(int i=0;i<tensor.rows();i++){
        for(int j=0;j<tensor.cols();j++){
                LOG(INFO)<<"tensor["<<i<<","<<j<<","<<"]:"<<tensor.at(tensor.channels()-1,i,j);
        }
    }

    std::vector<float> f_tensor={1,2,3,4,5,6,7,8,9,10,11,12};
    cel::Tensor<float> tensor_3d(f_tensor.data(),{2,3,2});
    // 1 4
    // 2 5
    // 3 6

    // 7 10
    // 8 11
    // 9 12

    LOG(INFO)<<"rows: "<<tensor_3d.rows();
    LOG(INFO)<<"cols: "<<tensor_3d.cols();
    LOG(INFO)<<"slices: "<<tensor_3d.channels();
    for(int i=0;i<12;i++){
        LOG(INFO)<<"tensor["<<i<<"]:"<<tensor_3d.index(i);
    }
    for(int i=0;i<tensor_3d.channels();i++){
        for(int j=0;j<tensor_3d.rows();j++){
            for(int k=0;k<tensor_3d.cols();k++){
                LOG(INFO)<<"tensor["<<i<<","<<j<<","<<k<<"]:"<<tensor_3d.at(i,j,k);
            }
        }
    }

    float* cube_data=tensor_3d.raw_ptr();
    for(int i=0;i<12;i++){
        LOG(INFO)<<"cube_data["<<i<<"]:"<<cube_data[i];
    }

    // 移动指针读取cube_data
    for(int i=0;i<tensor_3d.channels();i++){
        for(int j=0;j<tensor_3d.rows();j++){
            for(int k=0;k<tensor_3d.cols();k++){
                LOG(INFO)<<"cube_data["<<i<<","<<j<<","<<k<<"]:"<<*(cube_data+i*tensor_3d.rows()*tensor_3d.cols()+j*tensor_3d.cols()+k);
            }
        }
    }
}