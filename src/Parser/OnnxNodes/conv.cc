#include "Parser/node.hpp"
#include "Parser/OnnxNodes/conv.hpp"
#include "Parser/Compute/conv.hpp"

void cel::ConvNode::forward(){
    LOG(INFO)<<name()<<" forward start";
    Attribute attr=attribute();
    std::vector<int64_t> pads;
   std::any pads_any=attr.find("pads");
   if(pads_any.has_value()){
         pads=std::any_cast<std::vector<int64_t>>(pads_any);
   }else{
         pads={0,0,0,0};
    }
     std::vector<int64_t> stride;
     std::any stride_any=attr.find("strides");
     if(stride_any.has_value()){
          stride=std::any_cast<std::vector<int64_t>>(stride_any);
     }else{
          stride={1,1};
     }
     std::vector<int64_t> dilation;
     std::any dilation_any=attr.find("dilations");
     if(dilation_any.has_value()){
          dilation=std::any_cast<std::vector<int64_t>>(dilation_any);
     }else{
          dilation={1,1};
     }
     std::vector<int64_t> kernel_shape;
     std::any kernel_shape_any=attr.find("kernel_shape");
     if(kernel_shape_any.has_value()){
          kernel_shape=std::any_cast<std::vector<int64_t>>(kernel_shape_any);
     }else{
          kernel_shape={0,0};
     }
     int64_t group=1;
     std::any group_any=attr.find("group");
     if(group_any.has_value()){
          group=std::any_cast<int64_t>(group_any);
     }

     edge_vec input_edges=inputs();
     edge_ptr input_edge=input_edges[0];
     edge_ptr kernel_edge=input_edges[1];
     edge_ptr bias_edge=std::make_shared<Edge>();
     if(input_edges.size()==3){
          bias_edge=input_edges[2];
     }

    tensor_vec<float> input=input_edge->data();
    tensor_vec<float> kernel=kernel_edge->data();
    tensor_vec<float> bias=bias_edge->data();
    tensor_vec<float> output;
    conv_compute<float>(input,kernel,bias,output,pads,stride,dilation,kernel_shape,group);

     for(auto& output_edge:outputs()){
          output_edge->set_data(output);
     }
    LOG(INFO)<<name()<<" forward done";
}