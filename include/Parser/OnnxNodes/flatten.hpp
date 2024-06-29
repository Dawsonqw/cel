#ifndef BASE_PARSER_FLATTEN_H
#define BASE_PARSER_FLATTEN_H
#include "Parser/node.hpp"

namespace cel{
    class FlattenNode: public Node{
        public:
            FlattenNode():Node(){}
            FlattenNode(const std::string& name):Node(name,"Flatten"){}
            FlattenNode(const std::string& name,const Attribute& attribute):Node(name,"Flatten",attribute){}
            FlattenNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"Flatten",attribute,inputs,outputs){}
            ~FlattenNode()=default;
            void forward();
    };
}

#endif