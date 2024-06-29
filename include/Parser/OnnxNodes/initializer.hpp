#ifndef BASE_PARSER_INITIALIZER_H
#define BASE_PARSER_INITIALIZER_H
#include "Parser/node.hpp"

namespace cel{
    class InitializerNode: public Node{
        public:
            InitializerNode():Node(){}
            InitializerNode(const std::string& name):Node(name,"Initializer"){}
            InitializerNode(const std::string& name,const Attribute& attribute):Node(name,"Initializer",attribute){}
            InitializerNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"Initializer",attribute,inputs,outputs){}
            ~InitializerNode()=default;
            void forward();
    };
}

#endif