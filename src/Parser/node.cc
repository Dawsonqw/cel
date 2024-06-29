#include "Parser/node.hpp"

void cel::Node::set_name(const std::string &name) 
{
    m_name=name;
}

const std::string &cel::Node::name() const
{
    return m_name;
}

const std::string &cel::Node::node_type() const
{
    return m_type;
}

void cel::Node::set_node_type(const std::string &type)
{
    m_type=type;
}

void cel::Node::set_attribute(const Attribute &attribute)
{
    m_attribute=attribute;
}

const cel::Attribute &cel::Node::attribute() const
{
    return m_attribute;
}

void cel::Node::add_edge_input(edge_ptr input)
{
    m_input_edges.push_back(input);
}

void cel::Node::add_edge_output(edge_ptr output)
{
    m_output_edges.push_back(output);
}

void cel::Node::set_edge_inputs(const cel::edge_vec &inputs)
{
    m_input_edges=inputs;
}

const cel::edge_vec &cel::Node::inputs() const
{
    return m_input_edges;
}

const int32_t cel::Node::get_input_edge_num() const { 
    return m_input_edges.size();
}

void cel::Node::set_edge_outputs(const cel::edge_vec &outputs) { m_output_edges = outputs; }

const cel::edge_vec &cel::Node::outputs() const
{
    return m_output_edges;
}

const int32_t cel::Node::get_output_edge_num() const { 
    return m_output_edges.size();
}

void cel::Node::forward() {
    LOG(INFO)<<"node type:"<<m_type<<" name:"<<m_name<<" forward";
}
