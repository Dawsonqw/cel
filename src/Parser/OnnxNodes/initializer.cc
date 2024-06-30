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
    if(shape_vec.size()==4){
        int32_t batch=shape_vec[0];
        int32_t channel=shape_vec[1];
        int32_t height=shape_vec[2];
        int32_t width=shape_vec[3];
        int32_t size=channel*height*width;
        LOG_IF(ERROR,data_vec.size()!=batch*size)<<"InitializerNode::forward: data size is not equal to shape size";
        for(int32_t i=0;i<batch;i++){
            std::vector<float> data_batch(data_vec.begin()+i*size,data_vec.begin()+(i+1)*size);
            Tensor<float> initializer_tensor(data_batch.data(),{channel,height,width});
            tensor.push_back(std::make_shared<Tensor<float>>(initializer_tensor));
        }
    }else{
        Tensor<float> initializer_tensor(data_vec.data(),shape_vec);
        tensor.push_back(std::make_shared<Tensor<float>>(initializer_tensor));
    }

    LOG_IF(ERROR,get_output_edge_num()!=1)<<"InitializerNode::forward: "<<name()<<" has "<<get_output_edge_num()<<" outputs";
    edge_ptr output=outputs()[0];
    output->set_data(tensor);
    LOG(INFO)<<name()<<" forward done";
}