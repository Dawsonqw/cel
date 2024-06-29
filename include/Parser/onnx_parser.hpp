#ifndef ONNX_PARSER_HPP
#define ONNX_PARSER_HPP

#include "onnx.pb.h"
#include "parser.hpp"
#include "model.hpp"
#include "edge.hpp"
#include "Parser/OnnxNodes/nodes.hpp"
#include <glog/logging.h>
#include <queue>

namespace cel{
    enum class OnnxType{
        FLOAT=1,
        UINT8=2,
        INT8=3,
        UINT16=4,
        INT16=5,
        INT32=6,
        INT64=7,
        STRING=8,
        BOOL=9,
        FLOAT16=10,
        DOUBLE=11,
        UINT32=12,
        UINT64=13,
        COMPLEX64=14,
        COMPLEX128=15,
        BFLOAT16=16
    };

    class OnnxParser: public Parser{    
        public:
            using Parser::Parser;

            ~OnnxParser()=default;

            OnnxParser(const OnnxParser&)=delete;
            OnnxParser& operator=(const OnnxParser&)=delete;

            void parse(Model* model) override;

            void parse_info(Model* model);

            void parse_inoutput(Model* model);

            void parse_initializer(Model* model);

            void parse_nodes(Model* model);

            void build_graph(Model* model);

            static std::shared_ptr<cel::Node> create_node(const std::string& node_type);
            static std::any parse_attribute(const onnx::AttributeProto& attr);
            static onnx::AttributeProto to_attribute(const std::string& name,const std::any& value);
            static std::string get_onnx_type(OnnxType type);
        private:
            onnx::ModelProto m_model;
            std::map<std::string,std::pair<std::vector<int32_t>,OnnxType>> m_input_info;
            std::map<std::string,std::pair<std::vector<int32_t>,OnnxType>> m_output_info;
    };
}

#endif