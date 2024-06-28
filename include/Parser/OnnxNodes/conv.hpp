#ifndef BASE_PARSER_CONV_H
#define BASE_PARSER_CONV_H
#include "Parser/node.hpp"

namespace cel{
    class ConvNode: public Node{
        public:
            ConvNode()=default;
            ConvNode(const std::string& name):Node(name,"Conv"){}
            ConvNode(const std::string& name,const Attribute& attribute):Node(name,"Conv",attribute){}
            ConvNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"Conv",attribute,inputs,outputs){}
            virtual ~ConvNode()=default;
    };
}

#endif