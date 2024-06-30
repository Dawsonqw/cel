#include <gtest/gtest.h>
#include <string>
#include <glog/logging.h>
#include "Parser/tensor.hpp"
#include <armadillo>
#include <vector>


TEST(Arma, test_arma){
    google::InitGoogleLogging("test_arma");
    google::SetStderrLogging(google::INFO);
}