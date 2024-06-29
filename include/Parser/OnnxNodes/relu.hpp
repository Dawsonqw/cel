#ifndef BASE_PARSER_RELU_H
#define BASE_PARSER_RELU_H
#include "Parser/node.hpp"

namespace cel{
    class ReluNode: public Node{
        public:
            ReluNode():Node(){}
            ReluNode(const std::string& name):Node(name,"Relu"){}
            ReluNode(const std::string& name,const Attribute& attribute):Node(name,"Relu",attribute){}
            ReluNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"Relu",attribute,inputs,outputs){}
            ~ReluNode()=default;
            void forward();
    };
}

#endif