#include "Parser/node.hpp"
#include "Parser/OnnxNodes/maxpool.hpp"
#include "Parser/Compute/maxpool.hpp"


void cel::MaxPoolNode::forward(){
    LOG(INFO)<<name()<<" forward start";
    Attribute attr=attribute();
    int64_t ceil_mode=0;
    std::any ceil_mode_any=attr.find("ceil_mode");
    if(ceil_mode_any.has_value()){
        ceil_mode=std::any_cast<int64_t>(ceil_mode_any);
    }
    std::vector<int64_t>dilations;
    std::any dilations_any=attr.find("dilations");
    if(dilations_any.has_value()){
        dilations=std::any_cast<std::vector<int64_t>>(dilations_any);
    }else{
        dilations={1,1};
    }
    std::vector<int64_t>kernel_shape;
    std::any kernel_shape_any=attr.find("kernel_shape");
    if(kernel_shape_any.has_value()){
        kernel_shape=std::any_cast<std::vector<int64_t>>(kernel_shape_any);
    }else{
        kernel_shape={0,0};
    }
    std::vector<int64_t>pads;
    std::any pads_any=attr.find("pads");
    if(pads_any.has_value()){
        pads=std::any_cast<std::vector<int64_t>>(pads_any);
    }else{
        pads={0,0,0,0};
    }
    std::vector<int64_t>strides;
    std::any strides_any=attr.find("strides");
    if(strides_any.has_value()){
        strides=std::any_cast<std::vector<int64_t>>(strides_any);
    }else{
        strides={1,1};
    }
    edge_vec input_edges=inputs();
    CHECK_EQ(input_edges.size(),1)<<"MaxPool node should have 1 input edge";
    edge_ptr input_edge=input_edges[0];
    tensor_vec<float> input=input_edge->data();
    tensor_vec<float> output;
    maxpool_compute<float>(input,output,ceil_mode,kernel_shape,pads,strides,dilations);
    for(auto& output_edge:outputs()){
        output_edge->set_data(output);
    }
    LOG(INFO)<<name()<<" forward done";
}