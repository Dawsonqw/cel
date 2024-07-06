#include "Parser/node.hpp"
#include "Parser/OnnxNodes/globalaveragepool.hpp"
#include "Parser/Compute/globalaveragepool.hpp"

void cel::GlobalAveragePoolNode::forward(){
    LOG(INFO)<<name()<<" forward start";
    edge_vec input_edges=inputs();
    CHECK_EQ(input_edges.size(),1)<<"GlobalAveragePool node should have 1 input edge";
    edge_ptr input_edge=input_edges[0];
    tensor_vec<float> input=input_edge->data();
    tensor_vec<float> output;
    globalaveragepool_compute<float>(input,output);
    for(auto& output_edge:outputs()){
        output_edge->set_data(output);
    }
    LOG(INFO)<<name()<<" forward done";
}