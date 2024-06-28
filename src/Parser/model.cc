#include "Parser/model.hpp"

void cel::Model::set_name(const std::string &name)
{
    m_name=name;
}
const std::string &cel::Model::name() const
{
    return m_name;
}

void cel::Model::set_attribute(const Attribute &attribute)
{
    m_attribute = attribute;
}

const cel::Attribute &cel::Model::attribute() const
{
    return m_attribute;
}

bool cel::Model::add_node(const std::string &name, node_ptr node)
{
    if (m_node_map.find(name) != m_node_map.end())
    {
        return false;
    }
    m_node_map[name] = node;
    return true;
}

bool cel::Model::add_edge(const std::string &name, edge_ptr edge)
{
    if (m_edge_map.find(name) != m_edge_map.end())
    {
        return false;
    }
    m_edge_map[name] = edge;
    return true;
}

bool cel::Model::del_node(const std::string &name)
{
    if (m_node_map.find(name) == m_node_map.end())
    {
        return false;
    }
    m_node_map.erase(name);
    return true;
}

bool cel::Model::del_edge(const std::string &name)
{
    if (m_edge_map.find(name) == m_edge_map.end())
    {
        return false;
    }
    m_edge_map.erase(name);
    return true;
}

bool cel::Model::update_node(const std::string &name, node_ptr node)
{
    if (m_node_map.find(name) == m_node_map.end())
    {
        return false;
    }
    m_node_map[name] = node;
    return true;
}

bool cel::Model::update_edge(const std::string &name, edge_ptr edge)
{
    if (m_edge_map.find(name) == m_edge_map.end())
    {
        return false;
    }
    m_edge_map[name] = edge;
    return true;
}

bool cel::Model::add_input(edge_ptr input,int insert_point)
{
    if(insert_point==-1){
        m_model_inputs.push_back(input);
        return true;
    }
    if (insert_point > m_model_inputs.size() || std::find(m_model_inputs.begin(), m_model_inputs.end(), input) != m_model_inputs.end())
    {
        return false;
    }
    m_model_inputs.insert(m_model_inputs.begin() + insert_point, input);
    return true;
}

bool cel::Model::add_output(edge_ptr output,int insert_point)
{
    if(insert_point==-1){
        m_model_outputs.push_back(output);
        return true;
    }
    if (insert_point > m_model_outputs.size() || std::find(m_model_outputs.begin(), m_model_outputs.end(), output) != m_model_outputs.end())
    {
        return false;
    }
    m_model_outputs.insert(m_model_outputs.begin() + insert_point, output);
    return true;
}

bool cel::Model::del_input(const std::string &name)
{
    for (auto iter = m_model_inputs.begin(); iter != m_model_inputs.end(); ++iter)
    {
    }
    return true;
}

bool cel::Model::del_output(const std::string &name)
{
    for (auto iter = m_model_outputs.begin(); iter != m_model_outputs.end(); ++iter)
    {
    }
    return true;
}

void cel::Model::set_inputs(const edge_vec &inputs)
{
    m_model_inputs=inputs;
}

const cel::edge_vec &cel::Model::inputs() const
{
    return m_model_inputs;
}

void cel::Model::set_outputs(const edge_vec &outputs)
{
    m_model_outputs=outputs;
}

const cel::edge_vec &cel::Model::outputs() const
{
    return m_model_outputs;
}
