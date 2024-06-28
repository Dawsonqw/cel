#ifndef BASE_PARSER_GLOBALAVERAGEPOOL_H
#define BASE_PARSER_GLOBALAVERAGEPOOL_H
#include "Parser/node.hpp"

namespace cel{
    class GlobalAveragePoolNode: public Node{
        public:
            GlobalAveragePoolNode()=default;
            GlobalAveragePoolNode(const std::string& name):Node(name,"GlobalAveragePool"){}
            GlobalAveragePoolNode(const std::string& name,const Attribute& attribute):Node(name,"GlobalAveragePool",attribute){}
            GlobalAveragePoolNode(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :Node(name,"GlobalAveragePool",attribute,inputs,outputs){}
            virtual ~GlobalAveragePoolNode()=default;
    };
}

#endif