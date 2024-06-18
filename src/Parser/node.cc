#include "Parser/node.hpp"

void cel::Node::set_name(const std::string &name)
{
    m_name=name;
}

const std::string &cel::Node::name() const
{
    return m_name;
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

void cel::Node::set_edge_outputs(const cel::edge_vec &outputs)
{
    m_output_edges=outputs;
}
const cel::edge_vec &cel::Node::outputs() const
{
    return m_output_edges;
}
