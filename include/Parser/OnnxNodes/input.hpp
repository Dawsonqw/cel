#ifndef BASE_PARSER_INPUT_H
#define BASE_PARSER_INPUT_H
#include "Parser/node.hpp"

namespace cel{
    class InputNode: public Node{
        public:
            InputNode():Node(){}
            InputNode(const std::string& name):Node(name,"Input"){}
            InputNode(const std::string& name,const Attribute& attribute):Node(name,"Input",attribute){}
            InputNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"Input",attribute,inputs,outputs){}
            ~InputNode()=default;
            void forward();
    };
}

#endif