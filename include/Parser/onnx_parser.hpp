#ifndef ONNX_PARSER_HPP
#define ONNX_PARSER_HPP

#include "onnx.pb.h"
#include "parser.hpp"
#include "model.hpp"

namespace cel{
    class OnnxParser: public Parser{    
        public:
            using Parser::Parser;

            ~OnnxParser()=default;

            OnnxParser(const OnnxParser&)=delete;
            OnnxParser& operator=(const OnnxParser&)=delete;

            void parse() override;

        private:
            onnx::ModelProto m_model;
            Model m_cel_model;
    };
}

#endif