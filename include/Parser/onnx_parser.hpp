#ifndef ONNX_PARSER_HPP
#define ONNX_PARSER_HPP

#include "onnx.pb.h"
#include "parser.hpp"

namespace cel{
    class OnnxParser: public Parser{    
        public:
            OnnxParser(const std::string& file_name):Parser(file_name){}
            ~OnnxParser()=default;

            OnnxParser(const OnnxParser&)=delete;
            OnnxParser& operator=(const OnnxParser&)=delete;

            void parse() override;

        private:
            onnx::ModelProto m_model;
    };
}

#endif