#include "Parser/node.hpp"
#include "Parser/OnnxNodes/flatten.hpp"
#include "Parser/Compute/flatten.hpp"

void cel::FlattenNode::forward(){
    LOG(INFO)<<name()<<" forward start";
    Attribute attr=attribute();
    int64_t axis=1;
    std::any axis_any=attr.find("axis");
    if(axis_any.has_value()){
        axis=std::any_cast<int64_t>(axis_any);
    }
    edge_vec input_edges=inputs();
    CHECK_EQ(input_edges.size(),1)<<"Flatten node should have 1 input edge";
    edge_ptr input_edge=input_edges[0];
    tensor_vec<float> input=input_edge->data();
    tensor_vec<float> output;
    flatten_compute<float>(input,output,axis);
    for(auto& output_edge:outputs()){
        output_edge->set_data(output);
    }
    LOG(INFO)<<name()<<" forward done";
}