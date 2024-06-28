#ifndef BASE_PARSER_OUTPUT_H
#define BASE_PARSER_OUTPUT_H
#include "Parser/node.hpp"

namespace cel{
    class OutputNode: public Node{
        public:
            OutputNode()=default;
            OutputNode(const std::string& name):Node(name,"output"){}
            OutputNode(const std::string& name,const Attribute& attribute):Node(name,"output",attribute){}
            OutputNode(const std::string& name,const Attribute& attribute,const edge_vec& OutputNodes,const edge_vec& outputs)
                :Node(name,"output",attribute,OutputNodes,outputs){}
            virtual ~OutputNode()=default;
    };
}

#endif