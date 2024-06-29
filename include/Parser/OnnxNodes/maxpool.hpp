#ifndef BASE_PARSER_MAXPOOL_H
#define BASE_PARSER_MAXPOOL_H
#include "Parser/node.hpp"

namespace cel{
    class MaxPoolNode: public Node{
        public:
            MaxPoolNode():Node(){}
            MaxPoolNode(const std::string& name):Node(name,"MaxPool"){}
            MaxPoolNode(const std::string& name,const Attribute& attribute):Node(name,"MaxPool",attribute){}
            MaxPoolNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"MaxPool",attribute,inputs,outputs){}
            ~MaxPoolNode()=default;
            void forward();
    };
}

#endif