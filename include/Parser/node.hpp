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
            Node()=default;
            Node(const std::string& name):m_name(name),m_attribute(cel::Attribute()),m_input_edges(),m_output_edges(){}
            Node(const std::string& name,const Attribute& attribute):m_name(name),m_attribute(attribute),m_input_edges(),m_output_edges(){}
            Node(const std::string& name,const Attribute& attribute,const edge_vec& inputs,const edge_vec& outputs)
                                    :m_name(name),m_attribute(attribute),m_input_edges(inputs),m_output_edges(outputs){}

            virtual ~Node()=default;

            void set_name(const std::string& name);
            const std::string& name() const;

            void set_attribute(const Attribute& attribute);
            const Attribute& attribute() const;

            void add_edge_input(edge_ptr input);
            void add_edge_output(edge_ptr output);

            void set_edge_inputs(const edge_vec& inputs);
            const edge_vec& inputs() const;
            void set_edge_outputs(const edge_vec& outputs);
            const edge_vec& outputs() const;

        private:
            std::string m_name;
            Attribute m_attribute;
            edge_vec m_input_edges;
            edge_vec m_output_edges;
    };
}

#endif