#ifndef BASE_EDGE_H
#define BASE_EDGE_H
#include <memory>
#include "utils.hpp"

namespace cel{
    class Node;
    class Edge{
        public:
            Edge()=delete;
            Edge(const std::string& index):m_index(index),m_src(nullptr),m_input_index(0),m_dst(nullptr),m_output_index(0){}
            Edge(const std::string& index,node_ptr src,uint32_t input_index,node_ptr dst,uint32_t output_index)
                :m_index(index),m_src(src),m_input_index(input_index),m_dst(dst),m_output_index(output_index){}
            ~Edge()=default;

            node_ptr src() const;
            uint32_t input_index() const;
            node_ptr dst() const;
            uint32_t output_index() const;
            std::string index() const;

            void set_src(node_ptr src);
            void set_input_index(uint32_t input_index);
            void set_dst(node_ptr dst);
            void set_output_index(uint32_t output_index);
            void set_index(const std::string& index);

        private:
            std::string m_index;
            node_ptr m_src;
            uint32_t m_input_index;
            node_ptr m_dst;
            uint32_t m_output_index;
    };
}
#endif