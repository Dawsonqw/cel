#ifndef BASE_MODEL_H
#define BASE_MODEL_H
#include "node.hpp"
#include "edge.hpp"
#include "utils.hpp"
#include "tensor.hpp"
#include "tensor_utils.hpp"
#include <glog/logging.h>
#include <queue>

namespace cel{
    class Model{
        public:
            Model()=default;
            Model(const std::string& name):m_name(name),m_node_map(),m_edge_map(),m_attribute(){}
            Model(const std::string& name,const Attribute& attribute):m_name(name),m_node_map(),m_edge_map(),m_attribute(attribute){}
            Model(const std::string& name,const Attribute& attribute,const edge_map_t& inputs,const edge_map_t& outputs)
                :m_name(name),m_node_map(),m_edge_map(),m_attribute(attribute),m_model_inputs(inputs),m_model_outputs(outputs){}
            virtual ~Model()=default;

            void set_name(const std::string& name);
            const std::string& name() const;

            void set_attribute(const Attribute& attribute);
            const Attribute& attribute() const;

            bool is_nodeExist(const std::string& name) const;
            bool is_edgeEixst(const std::string& name) const;
            bool is_link_edgeExist(const std::string&name,node_ptr src_node,uint32_t src_index,node_ptr dst_node,uint32_t dst_index) const;
            
            edge_ptr get_link_edge(const std::string& name,node_ptr src_node,uint32_t src_index,node_ptr dst_node,uint32_t dst_index) const;

            int32_t get_node_num() const;
            int32_t get_edge_num() const;

            node_ptr get_node(const std::string& name) const;
            edge_vec get_edge(const std::string& name) const;

            bool add_node(const std::string& name,node_ptr node);
            bool add_edge(const std::string& name,edge_ptr edge);

            bool del_node(const std::string& name);
            bool del_edge(const std::string& name);

            bool update_node(const std::string& name,node_ptr node);
            bool update_edge(const std::string& name,edge_vec edge);

            bool add_input(const std::string& name,edge_ptr input);
            bool add_output(const std::string& name,edge_ptr output);

            bool del_input(const std::string& name);
            bool del_output(const std::string& name);

            void set_inputs(const edge_map_t& inputs);
            const edge_ptr& input(const std::string& name) const;
            const edge_map_t& inputs() const;

            void set_outputs(const edge_map_t& outputs);
            const edge_ptr& outputs(const std::string& name) const;
            const edge_map_t& outputs() const;

            bool verify() const;
                
            void topological_sort();

            void forward(std::map<std::string,tensor_vec<float>>& tensors);

        private:
            node_map_t m_node_map;
            edge_map m_edge_map;
            std::string m_name;
            Attribute m_attribute;
            edge_map_t m_model_inputs;
            edge_map_t m_model_outputs;
            std::queue<node_ptr> m_topo_seq;
    };
}

#endif