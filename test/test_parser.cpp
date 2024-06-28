#include "Parser/onnx_parser.hpp"
#include <string>
#include <glog/logging.h>

int main(int argc, char* argv[]){
    google::InitGoogleLogging(argv[0]); 
    LOG(INFO) << "Found " << 1 << " cookies";
    return 0;
}