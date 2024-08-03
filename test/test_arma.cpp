#include <gtest/gtest.h>
#include <string>
#include <glog/logging.h>
#include "Parser/tensor.hpp"
#include <armadillo>
#include <vector>
#include <iostream>

TEST(Arma, matmul_flatten){
    google::InitGoogleLogging("test_arma");
    google::SetStderrLogging(google::INFO);

    // 定义matmul
    std::vector<float>data={1,2,3,4,5,6,7,8,9,10,11,12};
    std::vector<int>shape={3,4};

    // copy_aux_mem为true表示构造对象时数据独立，arma进行数据拷贝；为false表示数据共享，直接使用对应的内存；
    // strict为true表示表示在分配内存或进行操作时，会严格遵守给定的维度和大小。如果要求的尺寸不匹配或超出范围，将引发错误;为false时不会严格检查尺寸，可能导致错误
    arma::Mat<float> matmul(data.data(),shape[0],shape[1],false,false);

    std::cout<<"============{3,4}===================\n";
    // print()函数用于输出到控制台
    matmul.print();

    // reshape函数用于改变矩阵的形状
    matmul.reshape(1,12);
    std::cout<<"============={1,12}==================\n";
    matmul.print();

    std::cout<<"============={4,3}==================\n";
    matmul.reshape(4,3);
    matmul.print();
}

TEST(Arma, matmul_matmul){
    std::vector<float>data1={1,2,3,4,5,6,7,8,9,10,11,12};
    std::vector<int>shape1={3,4};
    std::vector<float>data2={10,20,30,40,50,60,70,80,90,100,110,120};
    std::vector<int>shape2={4,3};

    arma::Mat<float> matmul1(data1.data(),shape1[0],shape1[1],false,false);
    arma::Mat<float> matmul2(data2.data(),shape2[0],shape2[1],false,false);

    std::cout<<"========= A ===========\n";
    matmul1.print();
    std::cout<<"========= B ===========\n";
    matmul2.print();
    arma::Mat<float> result=matmul1*matmul2;
    std::cout<<"A matmul B: \n";
    result.print();

    // 假设numpy数据是3*4的数组，数据为1-12，存储时按照行优先存储，由于arma是按照列优先存储
    std::vector<float>data={1,2,3,4,5,6,7,8,9,10,11,12};
    // 1 2 3 4
    // 5 6 7 8
    // 9 10 11 12
    std::vector<int>shape={4,3};
    arma::Mat<float> matmul(data.data(),shape[0],shape[1],false,false);
    std::cout<<"========= A ===========\n";
    matmul.print();
    std::vector<float>data3={1,2,3,4,5,6,7,8};
    // 1 2
    // 3 4
    // 5 6
    // 7 8
    arma::Mat<float> matmul3(data3.data(),2,4,false,false);
    std::cout<<"========= B ===========\n";
    matmul3.print();
    result=matmul3*matmul;
    std::cout<<"A matmul B: \n";
    result.print();
}

TEST(Arma,matmul_transpose){
    std::vector<float>data1={1,2,3,4,5,6,7,8,9,10,11,12};
    std::vector<int>shape1={3,4};

    arma::Mat<float> matmul1(data1.data(),shape1[0],shape1[1],false,false);

    std::cout<<"========= A ===========\n";
    matmul1.print();

    arma::Mat<float> result=matmul1.t();

    std::cout<<"A transpose: \n";
    result.print();
}


TEST(Arma,cube_construct){
    std::vector<float>data({1,2,3,4,5,6,7,8,9,10,11,12});
    std::vector<int>shape({2,3,2});
    arma::Cube<float> cube(data.data(),shape[0],shape[1],shape[2],false,false);
    std::cout<<"========= all ===========\n";
    cube.print();
    std::cout<<"========= raw ===========\n";
    cube.raw_print();
    std::cout<<"========= brief ===========\n";
    cube.brief_print();
}


TEST(Arma,cube_add){
    std::vector<float>data1({1,2,3,4,5,6,7,8,9,10,11,12});
    std::vector<int>shape1({2,3,2});
    std::vector<float>data2({10,20,30,40,50,60,70,80,90,100,110,120});
    std::vector<int>shape2({2,3,2});

    arma::Cube<float> cube1(data1.data(),shape1[0],shape1[1],shape1[2],false,false);
    arma::Cube<float> cube2(data2.data(),shape2[0],shape2[1],shape2[2],false,false);

    std::cout<<"========= A ===========\n";
    cube1.print();
    std::cout<<"========= B ===========\n";
    cube2.print();
    arma::Cube<float> result=cube1+cube2;
    std::cout<<"A matmul B: \n";
    result.print();
}


TEST(Arma,cube_matmul){
    std::vector<float>data1({1,2,3,4,5,6,7,8,9,10,11,12});
    std::vector<int>shape1({3,2,2});
    std::vector<float>data2({10,20,30,40,50,60,70,80,90,100,110,120});
    std::vector<int>shape2({2,2,3});

    arma::Cube<float> cube1(data1.data(),shape1[2],shape1[1],shape1[0],false,false);
    arma::Cube<float> cube2(data2.data(),shape2[2],shape2[1],shape2[0],false,false);

    std::cout<<"========= A ===========\n";
    cube1.brief_print();
    std::cout<<"========= B ===========\n";
    cube2.brief_print();
    arma::Cube<float> result(cube2.n_rows,cube1.n_cols,cube1.n_slices);
    for(int i=0;i<cube1.n_slices;i++){
        for(int j=0;j<cube2.n_slices;j++){
            result.slice(i)+=cube2.slice(j)*cube1.slice(i);
        }
    }
    std::cout<<"A matmul B: \n";
    std::cout<<"channel: "<<result.n_slices<<", row: "<<result.n_rows<<", col: "<<result.n_cols<<std::endl;
    result.brief_print();
}