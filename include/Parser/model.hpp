#ifndef BASE_MODEL_H
#define BASE_MODEL_H
#include "node.hpp"
#include "edge.hpp"
#include "utils.hpp"

namespace cel{
    class Model{
        public:
            using node_map=std::map<std::string,node_ptr>;
            using edge_map=std::map<std::string,edge_ptr>;
        public:
        private:
            node_map m_node_map;
            edge_map m_edge_map;
            std::string m_name;
            
    };
}

#endif