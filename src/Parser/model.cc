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

bool cel::Model::is_nodeExist(const std::string &name) const { 
    return m_node_map.find(name) != m_node_map.end();
}

bool cel::Model::is_edgeEixst(const std::string &name) const { 
    return m_edge_map.find(name) != m_edge_map.end();
}

int32_t cel::Model::get_node_num() const { 
    return m_node_map.size();
}

int32_t cel::Model::get_edge_num() const { 
    return m_edge_map.size();
}

cel::node_ptr cel::Model::get_node(const std::string &name) const {
    if (m_node_map.find(name) == m_node_map.end())
    {
        return node_ptr();
    }
    return m_node_map.at(name);
}

cel::edge_ptr cel::Model::get_edge(const std::string &name) const { 
    if(m_edge_map.find(name)==m_edge_map.end()){
        return edge_ptr();
    }
    return m_edge_map.at(name);
}

bool cel::Model::add_node(const std::string &name, node_ptr node) {

    if (m_node_map.find(name) != m_node_map.end()) {
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

void cel::Model::build_graph() {
}

bool cel::Model::verify() const {
    bool flag=true;
    for(auto iter=m_edge_map.begin();iter!=m_edge_map.end();++iter){
        if(iter->second->src()==nullptr || iter->second->dst()==nullptr){
            LOG(ERROR)<<"Edge "<<iter->first<<" is invalid";
            flag=false;
        }
    }
    for(auto iter=m_node_map.begin();iter!=m_node_map.end();++iter){
        if(iter->second==nullptr){
            LOG(ERROR)<<"Node "<<iter->first<<" is invalid";
            flag=false;
        }
    }
    return flag;
}

void cel::Model::topological_sort() {
    std::queue<node_ptr> topo_seq;
    std::map<node_ptr, int> in_degree;
    DLOG(INFO)<<"m_node_map size:"<<m_node_map.size();
    DLOG(INFO)<<"m_edge_map size:"<<m_edge_map.size();

    for (auto iter = m_node_map.begin(); iter != m_node_map.end(); ++iter)
    {
        in_degree[iter->second] = 0;
    }
    for (auto iter = m_edge_map.begin(); iter != m_edge_map.end(); ++iter)
    {
        in_degree[iter->second->dst()]++;
    }
    for (auto iter = in_degree.begin(); iter != in_degree.end(); ++iter)
    {
        if (iter->second == 0)
        {
            topo_seq.push(iter->first);
        }
    }
    DLOG(INFO)<<"Topological sort start,and the sequence size is:"<<topo_seq.size();
    while (!topo_seq.empty())
    {
        node_ptr node = topo_seq.front();
        m_topo_seq.push(node);
        topo_seq.pop();
        for (auto iter = node->outputs().begin(); iter != node->outputs().end(); ++iter)
        {
            DLOG(INFO)<<(*iter)->dst()->name()<<" in_degree: "<<in_degree[(*iter)->dst()];
            in_degree[(*iter)->dst()]--;
            if (in_degree[(*iter)->dst()] == 0)
            {
                topo_seq.push((*iter)->dst());
            }
        }
    }

    for(auto iter=in_degree.begin();iter!=in_degree.end();++iter){
        if(iter->second!=0){
            LOG(ERROR)<<"The model has a cycle";
            return;
        }
    }

    // 遍历m_topo_seq
    // for(int i=0;i<m_topo_seq.size();i++){
    //     DLOG(INFO)<<"Topological sort sequence:"<<m_topo_seq.front()->name();
    //     m_topo_seq.pop();
    // }
    LOG(INFO)<<"Topological sort done,and the sequence size is:"<<topo_seq.size();
}

void cel::Model::forward() {
    this->topological_sort();
}
