#ifndef BASE_PARSER_INPUT_H
#define BASE_PARSER_INPUT_H
#include "Parser/node.hpp"

namespace cel{
    class InputNode: public Node{
        public:
            InputNode()=default;
            InputNode(const std::string& name):Node(name,"input"){}
            InputNode(const std::string& name,const Attribute& attribute):Node(name,"input",attribute){}
            InputNode(const std::string& name,const Attribute& attribute,const edge_vec& InputNodes,const edge_vec& outputs)
                :Node(name,"input",attribute,InputNodes,outputs){}
            virtual ~InputNode()=default;
    };
}

#endif