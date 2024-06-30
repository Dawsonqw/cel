#include "Parser/edge.hpp"

cel::node_ptr cel::Edge::src() const
{
    return m_src;
}

uint32_t cel::Edge::input_index() const
{
    return m_input_index;

}

cel::node_ptr cel::Edge::dst() const
{
    return m_dst;
}

uint32_t cel::Edge::output_index() const
{
    return m_output_index;
}

std::string cel::Edge::index() const
{
    return m_index;
}

void cel::Edge::set_src(node_ptr src) { 
    m_src = src; 
}

void cel::Edge::set_input_index(uint32_t input_index)
{
    m_input_index=input_index;
}

void cel::Edge::set_dst(node_ptr dst)
{
    m_dst=dst;
}

void cel::Edge::set_output_index(uint32_t output_index)
{
    m_output_index=output_index;
}

void cel::Edge::set_index(const std::string &index)
{
    m_index=index;
}

void cel::Edge::set_data(const tensor_vec<float> &data_ptr) {
    m_data = data_ptr;
}

bool cel::Edge::has_data() const { 
    return !m_data.empty();
 }

void cel::Edge::release_data() {

}

const cel::tensor_vec<float> cel::Edge::data() const { 
    return m_data; 
}
