#ifndef BASE_NODE_H
#define BASE_NODE_H
#include <memory>
#include <vector>
#include "attribute.hpp"
#include "edge.hpp"
#include "utils.hpp"

namespace cel{
    class Node{
        public:
            Node():m_name(""),m_type(""),m_attribute(cel::Attribute()),m_input_edges(),m_output_edges(){}
            Node(const std::string& name,const std::string& type):m_name(name),m_type(type),m_attribute(cel::Attribute()),m_input_edges(),m_output_edges(){}
            Node(const std::string& name,const std::string& type,const Attribute& attribute):m_name(name),m_type(type),m_attribute(attribute),
                                    m_input_edges(),m_output_edges(){}
            Node(const std::string& name,const std::string& type,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                                    :m_name(name),m_type(type),m_attribute(attribute),m_input_edges(inputs),m_output_edges(outputs){}

            virtual ~Node(){}

            void set_name(const std::string& name);
            const std::string& name() const;

            void set_node_type(const std::string& type);
            const std::string& node_type()const;

            void set_attribute(const Attribute& attribute);
            const Attribute& attribute() const;

            void add_edge_input(edge_ptr input);
            void add_edge_output(edge_ptr output);

            void set_edge_inputs(const edge_vec& inputs);
            const edge_vec& inputs() const;
            const int32_t get_input_edge_num() const;

            void set_edge_outputs(const edge_vec& outputs);
            const edge_vec& outputs() const;
            const int32_t get_output_edge_num() const;

            virtual void forward();

        private:
            std::string m_name;
            std::string m_type;
            Attribute m_attribute;
            edge_vec m_input_edges;
            edge_vec m_output_edges;
    };
}

#endif