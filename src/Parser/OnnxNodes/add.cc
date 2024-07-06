#include "Parser/node.hpp"
#include "Parser/OnnxNodes/add.hpp"
#include "Parser/Compute/add.hpp"

void cel::AddNode::forward(){
    LOG(INFO)<<name()<<" forward start";
    edge_vec input_edges=inputs();
    CHECK_EQ(input_edges.size(),2)<<"Add node should have 2 input edge";
    edge_ptr input_edge_first=input_edges[0];
    edge_ptr input_edge_second=input_edges[1];
    tensor_vec<float> input_first=input_edge_first->data();
    tensor_vec<float> input_second=input_edge_second->data();
    tensor_vec<float> output;
    add_compute<float>(input_first,input_second,output);
    for(auto& output_edge:outputs()){
        output_edge->set_data(output);
    }
    LOG(INFO)<<name()<<" forward done";
}