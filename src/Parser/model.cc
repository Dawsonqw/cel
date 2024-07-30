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

bool cel::Model::is_link_edgeExist(const std::string &name, node_ptr src_node, uint32_t src_index,
                                   node_ptr dst_node, uint32_t dst_index) const {
    if (m_edge_map.find(name) == m_edge_map.end())
    {
        return false;
    }
    for (auto edge : m_edge_map.at(name))
    {
        if (edge->src() == src_node && edge->input_index() == src_index && edge->dst() == dst_node && edge->output_index() == dst_index)
        {
            return true;
        }
    }
    return false;
}

cel::edge_ptr cel::Model::get_link_edge(const std::string &name, node_ptr src_node, uint32_t src_index,
                                   node_ptr dst_node, uint32_t dst_index) const {
    if (m_edge_map.find(name) == m_edge_map.end())
    {
        return edge_ptr();
    }
    for (auto edge : m_edge_map.at(name))
    {
        if (edge->src() == src_node && edge->input_index() == src_index && edge->dst() == dst_node && edge->output_index() == dst_index)
        {
            return edge;
        }
    }
    return edge_ptr();
}

int32_t cel::Model::get_node_num() const { return m_node_map.size(); }

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

std::vector<cel::edge_ptr> cel::Model::get_edge(const std::string &name) const { 
    if(m_edge_map.find(name)==m_edge_map.end()){
        return std::vector<edge_ptr>();
    }
    std::vector<edge_ptr> edges=m_edge_map.at(name);
    return edges;
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
    if(m_edge_map[name].empty()){
        m_edge_map[name]=std::vector<edge_ptr>();
        m_edge_map[name].push_back(edge);
        return true;
    }
    for(auto target_edge:m_edge_map[name]){
        if(target_edge->src()==edge->src()&&target_edge->dst()==edge->dst()&&target_edge->input_index()==edge->input_index()&&target_edge->output_index()==edge->output_index()){
            return false;
        }
    }
    m_edge_map[name].push_back(edge);
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

bool cel::Model::update_edge(const std::string &name, edge_vec edge)
{
    if (m_edge_map.find(name) == m_edge_map.end())
    {
        return false;
    }
    m_edge_map[name] = edge;
    return true;
}

bool cel::Model::add_input(const std::string& name,edge_ptr input){
    if(m_model_inputs.find(name)!=m_model_inputs.end()){
        return false;
    }
    m_model_inputs[name]=input;
    return true;
}

bool cel::Model::add_output(const std::string& name,edge_ptr output){
    if(m_model_outputs.find(name)!=m_model_outputs.end()){
        return false;
    }
    m_model_outputs[name]=output;
    return true;
}

bool cel::Model::del_input(const std::string &name)
{
    if(m_model_inputs.find(name)==m_model_inputs.end()){
        return false;
    }
    m_model_inputs.erase(name);
}

bool cel::Model::del_output(const std::string &name)
{
    if(m_model_outputs.find(name)==m_model_outputs.end()){
        return false;
    }
    m_model_outputs.erase(name);
}

void cel::Model::set_inputs(const edge_map_t &inputs)
{
    m_model_inputs=inputs;
}

const cel::edge_map_t &cel::Model::inputs() const
{
    return m_model_inputs;
}

void cel::Model::set_outputs(const edge_map_t &outputs)
{
    m_model_outputs=outputs;
}

const cel::edge_map_t &cel::Model::outputs() const
{
    return m_model_outputs;
}

bool cel::Model::verify() const {
    bool flag=true;
    for(auto iter=m_edge_map.begin();iter!=m_edge_map.end();++iter){
        if(iter->second.empty()){
            LOG(ERROR)<<"Edge "<<iter->first<<" is invalid";
            flag=false;
        }
        for(auto edge:iter->second){
            if(edge==nullptr){
                LOG(ERROR)<<"Edge "<<iter->first<<" is invalid";
                flag=false;
            }
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
    std::map<node_ptr, int32_t> in_degree;
    DLOG(INFO)<<"m_node_map size:"<<m_node_map.size();
    DLOG(INFO)<<"m_edge_map size:"<<m_edge_map.size();

    for (auto iter = m_node_map.begin(); iter != m_node_map.end(); ++iter)
    {
        in_degree[iter->second] = iter->second->get_input_edge_num();
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
            node_ptr dst_node=(*iter)->dst();
            DLOG(INFO)<<dst_node->name()<<" in_degree: "<<in_degree[dst_node];
            in_degree[dst_node]--;
            if (in_degree[dst_node] == 0)
            {
                topo_seq.push(dst_node);
            }
        }
    }

    for(auto iter=in_degree.begin();iter!=in_degree.end();++iter){
        if(iter->second!=0){
            LOG(ERROR)<<"The model has a cycle";
            return;
        }
    }
    LOG(INFO)<<"Topological sort done";
}

void cel::Model::forward(std::map<std::string,tensor_vec<float>>& tensors){
    this->topological_sort();
    LOG(INFO)<<m_name<<" forward start";
    for(const auto& input_info:m_model_inputs){
        if(tensors.find(input_info.first)==tensors.end()){
            LOG(ERROR)<<"Input tensor "<<input_info.first<<" is not found";
            return;
        }
        for(auto& edge:m_edge_map[input_info.first]){
            edge->set_data(tensors[input_info.first]);
        }
    }
    while (!m_topo_seq.empty())
    {
        node_ptr node = m_topo_seq.front();
        node->forward();
        m_topo_seq.pop();
    }
}
