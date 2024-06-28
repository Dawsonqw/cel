#ifndef BASE_PARSER_ADD_H
#define BASE_PARSER_ADD_H
#include "Parser/node.hpp"

namespace cel{
    class AddNode: public Node{
        public:
            AddNode()=default;
            AddNode(const std::string& name):Node(name,"Add"){}
            AddNode(const std::string& name,const Attribute& attribute):Node(name,"Add",attribute){}
            AddNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"Add",attribute,inputs,outputs){}
            virtual ~AddNode()=default;
    };
}

#endif