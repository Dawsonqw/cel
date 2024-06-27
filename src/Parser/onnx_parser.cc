#include "Parser/onnx_parser.hpp"
#include <filesystem> 
// #include <glog/logging.h>

void cel::OnnxParser::parse()
{
    if (!std::filesystem::exists(m_filename)) {
        // LOG(ERROR)<<m_filename<<" does not exist!";
        return;
    }
    // 打开文件，以二进制写入的方式
    std::fstream input(m_filename, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!input.is_open()) {
        // LOG(ERROR)<<"Failed to open "<<m_filename;
        return;
    }

    // 获取文件大小
    std::streamsize size = input.tellg();
	input.seekg(0, std::ios::beg);
	std::vector<char> buffer(size);
	input.read(buffer.data(), size);

    m_model=onnx::ModelProto();
    m_model.ParseFromArray(buffer.data(), size);

    auto graph = m_model.graph();
}