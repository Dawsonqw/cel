#include "Parser/node.hpp"
#include "Parser/OnnxNodes/input.hpp"

void cel::InputNode::forward() {
    LOG(INFO)<<name()<<" forward start";
    LOG_IF(ERROR,get_output_edge_num()!=1)<<name()<<" has "<<get_output_edge_num()<<" inputs";
    bool is_ready=true;
    for(auto& output:outputs()){
        if(output->has_data()==false){
            is_ready=false;
            break;
        }
    }
    LOG_IF(ERROR,is_ready==false)<<"InputNode::forward: "<<name()<<" is not ready";
    LOG(INFO)<<name()<<" forward done";
}

void cel::InputNode::set_data(const std::string& name,const tensor_vec<float> &data) {
    edge_vec inputs=this->inputs();
    for(auto& input:inputs){
        if(input->index()==name){
            input->set_data(data);
            return;
        }
    }
    LOG(ERROR)<<"InputNode::set_data: Can't find the input edge named "<<name;
}