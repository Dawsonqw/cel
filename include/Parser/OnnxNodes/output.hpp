#ifndef BASE_PARSER_OUTPUT_H
#define BASE_PARSER_OUTPUT_H
#include "Parser/node.hpp"

namespace cel{
    class OutputNode: public Node{
        public:
            OutputNode():Node(){}
            OutputNode(const std::string& name):Node(name,"Output"){}
            OutputNode(const std::string& name,const Attribute& attribute):Node(name,"Output",attribute){}
            OutputNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"Output",attribute,inputs,outputs){}
            ~OutputNode()=default;
            void forward();
    };
}

#endif