#include "Parser/model.hpp"

void cel::Model::set_name(const std::string &name)
{
}
const std::string &cel::Model::name() const
{
    // TODO: 在此处插入 return 语句
}

void cel::Model::set_attribute(const Attribute &attribute)
{
}

const cel::Attribute &cel::Model::attribute() const
{
    // TODO: 在此处插入 return 语句
}

bool cel::Model::add_node(const std::string &name, node_ptr node)
{
}

bool cel::Model::add_edge(const std::string &name, edge_ptr edge)
{
}

bool cel::Model::del_node(const std::string &name)
{
    return false;
}

bool cel::Model::del_edge(const std::string &name)
{
    return false;
}

bool cel::Model::update_node(const std::string &name, node_ptr node)
{
    return false;
}

bool cel::Model::update_edge(const std::string &name, edge_ptr edge)
{
    return false;
}

bool cel::Model::add_input(edge_ptr input)
{
}

bool cel::Model::add_output(edge_ptr output)
{
}

bool cel::Model::del_input(const std::string &name)
{
    return false;
}

bool cel::Model::del_output(const std::string &name)
{
    return false;
}

void cel::Model::set_inputs(const edge_vec &inputs)
{
}

const cel::edge_vec &cel::Model::inputs() const
{
    // TODO: 在此处插入 return 语句
}

void cel::Model::set_outputs(const edge_vec &outputs)
{
}

const cel::edge_vec &cel::Model::outputs() const
{
    // TODO: 在此处插入 return 语句
}
