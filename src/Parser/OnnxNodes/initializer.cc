#include "Parser/node.hpp"
#include "Parser/OnnxNodes/initializer.hpp"
#include "Parser/tensor.hpp"
// #include "Parser/utils.hpp"

void cel::InitializerNode::forward(){
    LOG(INFO)<<name()<<" forward start";
    Attribute attribute=this->attribute();
    std::any data=attribute["data"];
    std::any shape=attribute["shape"];
    LOG_IF(ERROR,data.type()!=typeid(std::vector<float>))<<"InitializerNode::forward: data is not vector<float> type";
    LOG_IF(ERROR,shape.type()!=typeid(std::vector<int32_t>))<<"InitializerNode::forward: shape is not vector<int32_t> type";

    std::vector<float> data_vec=std::any_cast<std::vector<float>>(data);
    std::vector<int32_t> shape_vec=std::any_cast<std::vector<int32_t>>(shape);

    cel::tensor_vec<float> tensor;
    Tensor<float> initializer_tensor(data_vec.data(),shape_vec);
    tensor.push_back(std::make_shared<Tensor<float>>(initializer_tensor));

    LOG_IF(ERROR,get_output_edge_num()!=1)<<"InitializerNode::forward: "<<name()<<" has "<<get_output_edge_num()<<" outputs";
    edge_ptr output=outputs()[0];
    output->set_data(tensor);
    LOG(INFO)<<name()<<" forward done";
}