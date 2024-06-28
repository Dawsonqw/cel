#ifndef BASE_PARSER_INITIALIZER_H
#define BASE_PARSER_INITIALIZER_H
#include "Parser/node.hpp"

namespace cel{
    class InitializerNode: public Node{
        public:
            InitializerNode()=default;
            InitializerNode(const std::string& name):Node(name,"initializer"){}
            InitializerNode(const std::string& name,const Attribute& attribute):Node(name,"initializer",attribute){}
            InitializerNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"initializer",attribute,inputs,outputs){}
            virtual ~InitializerNode()=default;
    };
}

#endif