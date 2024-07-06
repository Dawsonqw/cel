#include "Parser/node.hpp"
#include "Parser/OnnxNodes/output.hpp"

void cel::OutputNode::forward(){
    LOG(INFO)<<name()<<" forward start";
    edge_vec input_edges=inputs();
    CHECK_EQ(input_edges.size(),1)<<"Output node should have 1 input edge";
    edge_ptr input_edge=input_edges[0];
    tensor_vec<float> input=input_edge->data();
    for(auto& output_edge:outputs()){
        output_edge->set_data(input);
    }
    LOG(INFO)<<name()<<" forward done";
}