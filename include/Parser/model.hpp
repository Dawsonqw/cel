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
            using node_map=std::map<std::string,node_ptr>;
            using edge_map=std::map<std::string,edge_vec>;
        public:
            Model()=default;
            Model(const std::string& name):m_name(name),m_node_map(),m_edge_map(),m_attribute(){}
            Model(const std::string& name,const Attribute& attribute):m_name(name),m_node_map(),m_edge_map(),m_attribute(attribute){}
            Model(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                :m_name(name),m_node_map(),m_edge_map(),m_attribute(attribute),m_model_inputs(inputs),m_model_outputs(outputs){}
            virtual ~Model()=default;

            void set_name(const std::string& name);
            const std::string& name() const;

            void set_attribute(const Attribute& attribute);
            const Attribute& attribute() const;

            bool is_nodeExist(const std::string& name) const;
            bool is_edgeEixst(const std::string& name) const;

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

            bool add_input(edge_ptr input,int insert_point=-1);
            bool add_output(edge_ptr output,int insert_point=-1);

            bool del_input(const std::string& name);
            bool del_output(const std::string& name);

            void set_inputs(const edge_vec& inputs);
            const edge_vec& inputs() const;

            void set_outputs(const edge_vec& outputs);
            const edge_vec& outputs() const;

            bool verify() const;
                
            void topological_sort();

            void forward(std::map<std::string,Tensor<float>>& tensors);
        private:
            node_map m_node_map;
            edge_map m_edge_map;
            std::string m_name;
            Attribute m_attribute;
            edge_vec m_model_inputs;
            edge_vec m_model_outputs;
            std::queue<node_ptr> m_topo_seq;
    };
}

#endif