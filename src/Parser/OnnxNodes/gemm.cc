#include "Parser/node.hpp"
#include "Parser/OnnxNodes/gemm.hpp"
#include "Parser/Compute/gemm.hpp"

void cel::GemmNode::forward(){
    LOG(INFO)<<name()<<" forward start";
    Attribute attr=attribute();
    float alpha=1.0f;
    float beta=1.0f;
    int64_t transB=1;
    std::any alpha_any=attr.find("alpha");
    if(alpha_any.has_value()){
        alpha=std::any_cast<float>(alpha_any);
    }
    std::any beta_any=attr.find("beta");
    if(beta_any.has_value()){
        beta=std::any_cast<float>(beta_any);
    }
    std::any transB_any=attr.find("transB");
    if(transB_any.has_value()){
        transB=std::any_cast<int64_t>(transB_any);
    }
    edge_vec input_edges=inputs();
    // 有3个或者2个输入
    CHECK(input_edges.size()==3||input_edges.size()==2)<<"Gemm node should have 2 or 3 input edge";
    edge_ptr input_edge=input_edges[0];
    edge_ptr weight_edge=input_edges[1];
    edge_ptr bias_edge=std::make_shared<Edge>();
    if(input_edges.size()==3){
        bias_edge=input_edges[2];
    }
    tensor_vec<float> input=input_edge->data();
    tensor_vec<float> weight=weight_edge->data();
    tensor_vec<float> bias=bias_edge->data();
    tensor_vec<float> output;
    gemm_compute<float>(input,weight,bias,output,alpha,beta,transB);
    for(auto& output_edge:outputs()){
        output_edge->set_data(output);
    }
    LOG(INFO)<<name()<<" forward done";
}