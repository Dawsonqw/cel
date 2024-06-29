#ifndef BASE_PARSER_GEMM_H
#define BASE_PARSER_GEMM_H
#include "Parser/node.hpp"

namespace cel{
    class GemmNode: public Node{
        public:
            GemmNode(){}
            GemmNode(const std::string& name):Node(name,"Gemm"){}
            GemmNode(const std::string& name,const Attribute& attribute):Node(name,"Gemm",attribute){}
            GemmNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"Gemm",attribute,inputs,outputs){}
            ~GemmNode()=default;
            void forward();
    };
}

#endif