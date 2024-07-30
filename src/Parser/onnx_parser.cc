#include "Parser/onnx_parser.hpp"
#include <algorithm>
#include <filesystem> 

void cel::OnnxParser::parse(Model* model)
{
    if(!std::filesystem::exists(m_filename))
    {
        LOG(ERROR)<<"File "<<m_filename<<" does not exist!";
        return;
    }
    std::ifstream file(m_filename,std::ios::binary);
    if(!file.is_open())
    {
        LOG(ERROR)<<"Failed to open file "<<m_filename;
        return;
    }
    if(!m_model.ParseFromIstream(&file))
    {
        LOG(ERROR)<<"Failed to parse onnx model from file "<<m_filename;
        return;
    }
    LOG(INFO)<<"Successfully parsed onnx model from file "<<m_filename;

    parse_info(model);
    parse_inoutput(model);
    parse_initializer(model);
    parse_edge(model);
    parse_nodes(model);
    build_graph(model);
}

void cel::OnnxParser::parse_info(Model *model) {
    Attribute attribute;
    if(m_model.has_producer_name())
    {
        std::string producer_name=m_model.producer_name();
        attribute.insert({"producer_name",producer_name});
        LOG(INFO)<<"Producer name: "<<producer_name;
    }
    if(m_model.has_producer_version())
    {
        std::string producer_version=m_model.producer_version();
        attribute.insert({"producer_version",producer_version});
        LOG(INFO)<<"Producer version: "<<producer_version;
    }
    if(m_model.has_domain())
    {
        std::string domain=m_model.domain();
        attribute.insert({"domain",domain});
        LOG(INFO)<<"Domain: "<<domain;
    }
    if(m_model.has_model_version())
    {
        int64_t model_version=m_model.model_version();
        attribute.insert({"model_version",model_version});
        LOG(INFO)<<"Model version: "<<model_version;
    }
    if(m_model.has_doc_string())
    {
        std::string doc_string=m_model.doc_string();
        attribute.insert({"doc_string",doc_string});
        LOG(INFO)<<"Doc string: "<<doc_string;
    }
    if(m_model.has_ir_version())
    {
        int64_t ir_version=m_model.ir_version();
        attribute.insert({"ir_version",ir_version});
        LOG(INFO)<<"IR version: "<<ir_version;
    }
    model->set_attribute(attribute);
}

void cel::OnnxParser::parse_inoutput(Model *model) {
    for (const auto& input : m_model.graph().input()) {
            node_ptr node=std::make_shared<InputNode>(input.name());
            LOG(INFO) << "Input name: " << input.name();

            Attribute attribute;

            std::string tensor_type="tensor_type";
            int32_t elem_type=input.type().tensor_type().elem_type();
            OnnxType onnx_type=static_cast<OnnxType>(elem_type);
            attribute.insert({tensor_type,onnx_type});
            
            std::string shape="shape";
            std::vector<int32_t> dims;
            if(input.type().tensor_type().has_shape())
            {
                for (const auto& dim : input.type().tensor_type().shape().dim()) {
                    dims.push_back(dim.dim_value());
                }
            }else{
                LOG(WARNING)<<"Input tensor "<<input.name()<<" has no shape!";
            }
            m_input_info.insert({input.name(),{dims,onnx_type}});

            attribute.insert({shape,dims});
            node->set_attribute(attribute);

            model->add_node(node->name(),node);
            LOG(INFO)<<node->name()<<" added to model";
    }

    for (const auto& output : m_model.graph().output()) {
        node_ptr node=std::make_shared<OutputNode>(output.name());
        LOG(INFO) << "Output name: " << output.name();

        Attribute attribute;

        std::string tensor_type="tensor_type";
        int32_t elem_type=output.type().tensor_type().elem_type();
        OnnxType onnx_type=static_cast<OnnxType>(elem_type);
        attribute.insert({tensor_type,onnx_type});
        
        std::string shape="shape";
        std::vector<int32_t> dims;
        if(output.type().tensor_type().has_shape())
        {
            for (const auto& dim : output.type().tensor_type().shape().dim()) {
                dims.push_back(dim.dim_value());
            }
        }else{
            LOG(WARNING)<<"Output tensor "<<output.name()<<" has no shape!";
        }
        m_output_info.insert({output.name(),{dims,onnx_type}});

        attribute.insert({shape,dims});
        node->set_attribute(attribute);

        model->add_node(node->name(),node);
        LOG(INFO)<<node->name()<<" added to model";
    }
}

void cel::OnnxParser::parse_initializer(Model* model){
    for (const auto& initializer : m_model.graph().initializer()) {
        node_ptr node=std::make_shared<InitializerNode>(initializer.name());
        LOG(INFO) << "Initializer name: " << initializer.name();

        Attribute attribute;

        std::string tensor_type="tensor_type";
        int32_t elem_type=initializer.data_type();
        OnnxType onnx_type=static_cast<OnnxType>(elem_type);
        attribute.insert({tensor_type,onnx_type});
        
        std::string shape="shape";
        std::vector<int32_t> dims;
        for (const auto& dim : initializer.dims()) {
            dims.push_back(dim);
        }
        attribute.insert({shape,dims});

        std::string data="data";
        if(onnx_type==OnnxType::FLOAT)
        {
            std::vector<float> data_vec=parse_tensor<float>(initializer,onnx_type);
            attribute.insert({data,data_vec});
        }else if(onnx_type==OnnxType::INT64)
        {
            std::vector<int64_t> data_vec=parse_tensor<int64_t>(initializer,onnx_type);
            attribute.insert({data,data_vec});
        }else if(onnx_type==OnnxType::INT32)
        {
            std::vector<int32_t> data_vec=parse_tensor<int32_t>(initializer,onnx_type);
            attribute.insert({data,data_vec});
        }else{
            LOG(WARNING)<<"Initializer "<<initializer.name()<<" has unsupported data type!";
        }
        node->set_attribute(attribute);

        model->add_node(node->name(),node);
        LOG(INFO)<<node->name()<<" added to model";
    }
}

void cel::OnnxParser::parse_edge(Model *model) {
    // input outut
    for (const auto& input : m_model.graph().input()) {
        m_edge_inputs.insert({input.name(),{std::pair<std::string,uint32_t>(input.name(),0)}});
    }
    for(const auto& output : m_model.graph().output())
    {
        m_edge_outputs.insert({output.name(),{std::pair<std::string,uint32_t>(output.name(),0)}});
    }

    // initializer
    for(const auto& initializer : m_model.graph().initializer())
    {
        m_edge_inputs.insert({initializer.name(),{std::pair<std::string,uint32_t>(initializer.name(),0)}});
    }

    // nodes
    for(const auto& node : m_model.graph().node())
    {
        int32_t  input_index=0;
        for(const auto& input : node.input())
        {
            if(m_edge_outputs.find(input)==m_edge_outputs.end())
            {
                m_edge_outputs.insert({input,{std::pair<std::string,uint32_t>(node.name(),input_index)}});
            }else{
                m_edge_outputs[input].push_back(std::pair<std::string,uint32_t>(node.name(),input_index));
            }
            input_index++;
        }

        int32_t output_index=0;
        for(const auto& output : node.output())
        {
            LOG_IF(FATAL,m_edge_inputs.find(output)!=m_edge_inputs.end())<<"Edge "<<output<<" has multiple inputs!";
            m_edge_inputs.insert({output,{std::pair<std::string,uint32_t>(node.name(),output_index)}});
            output_index++;
        }
    }
}

void cel::OnnxParser::parse_nodes(Model *model) {
    for (const auto& node : m_model.graph().node()) {
        std::string node_type=node.op_type();
        std::string node_name=node.name();
        node_ptr cur_node_ptr=create_node(node_type);
        if(cur_node_ptr==nullptr)
        {
            LOG(ERROR)<<"Node type "<<node_type<<" not supported!";
            continue;
        }
        cur_node_ptr->set_name(node_name);
        cur_node_ptr->set_node_type(node_type);
        LOG(INFO) << "Node name: " << node_name;
        LOG(INFO) << "Node type: " << node_type;

        Attribute attribute;
        for (const auto& attr : node.attribute()) {
            std::string attr_name=attr.name();
            std::any attr_value=parse_attribute(attr);
            attribute.insert({attr_name,attr_value});
        }

        cur_node_ptr->set_attribute(attribute);

        edge_vec inputs;
        uint32_t input_index=0;

        model->add_node(cur_node_ptr->name(),cur_node_ptr);
        LOG(INFO)<<"Node "<<cur_node_ptr->name()<<" added to model";
    }

    LOG_IF(FATAL,m_edge_inputs.size()!=m_edge_outputs.size())<<"Input edge num "<<m_edge_inputs.size()<<" not equal to output edge num "<<m_edge_outputs.size();

    for (const auto& node : m_model.graph().node()) {
        int32_t input_index=0;
        std::string node_name=node.name();
        LOG_IF(FATAL,model->get_node(node_name)==nullptr)<<"Node "<<node_name<<" not found!";
        node_ptr cur_node=model->get_node(node_name);
        edge_vec node_inputs;
        edge_vec node_outputs;
        for(const auto& input : node.input())
        {
            LOG_IF(FATAL,m_edge_inputs[input].size()!=1)<<"Edge "<<input<<" has multiple outputs!";
            std::pair<std::string,uint32_t> src_pair=m_edge_inputs[input][0];
            std::string src_node_name=src_pair.first;
            uint32_t src_index=src_pair.second;
            node_ptr src_node=model->get_node(src_node_name);
            LOG_IF(FATAL,src_node==nullptr)<<"Node "<<src_node_name<<" not found!";
            edge_ptr=model->get_link_edge(input,src_node,src_index,cur_node,input_index);
            LOG_IF(FATAL,edge==nullptr)<<"Edge "<<input<<" not found!";
            bool flag=model->add_edge(edge->index(),edge);
            LOG_IF(FATAL,flag==false)<<"Edge "<<input<<" existed!";
            node_inputs.push_back(edge);
        }
        cur_node->set_edge_inputs(node_inputs);
        
        int32_t output_index=0;
        for(const auto& output : node.output())
        {
            std::vector<std::pair<std::string,int32_t>> dst_pair_vec=m_edge_outputs[output];
            for(const auto& dst_pair : dst_pair_vec)
            {
                std::string dst_node_name=dst_pair.first;
                uint32_t dst_index=dst_pair.second;
                node_ptr dst_node=model->get_node(dst_node_name);
                LOG_IF(FATAL,dst_node==nullptr)<<"Node "<<dst_node_name<<" not found!";
                if(model->is_link_edgeExist(output,cur_node,input_index,dst_node,dst_index)==true)
                {
                    LOG(FATAL)<<"Edge "<<output<<" existed!";
                    // edge_ptr edge=model->get_link_edge(output,cur_node,input_index,dst_node,dst_index);
                    // if(edge!=nullptr)
                    // {
                    //     LOG(FATAL)<<output<<"not found!";
                    // }
                    // node_outputs.push_back(edge);
                }else{
                    edge_ptr edge=std::make_shared<Edge>(output,cur_node,output_index,dst_node,dst_index);
                    model->add_edge(edge->index(),edge);
                    node_outputs.push_back(edge);
                }
            }
        }
        cur_node->set_edge_outputs(node_outputs);
    }

    for (const auto& input : m_model.graph().input()) {
        node_ptr node=model->get_node(input.name());
        LOG_IF(FATAL,node==nullptr)<<"Node "<<input.name()<<" not found!";
        LOG_IF(FATAL,model->is_edgeEixst(input.name())==false)<<"Edge "<<input.name()<<" not found!";
        edge_vec outputs=model->get_edge(input.name());
        node->set_edge_outputs(outputs);
        LOG_IF(ERROR,outputs.size()!=1)<<input.name()<<" has "<<outputs.size()<<" outputs";
        model->add_input(input.name(),outputs[0]);
    }
    for(const auto& output : m_model.graph().output())
    {
        node_ptr node=model->get_node(output.name());
        LOG_IF(FATAL,node==nullptr)<<"Node "<<output.name()<<" not found!";
        LOG_IF(FATAL,model->is_edgeEixst(output.name())==false)<<"Edge "<<output.name()<<" not found!";
        edge_vec inputs=model->get_edge(output.name());
        node->set_edge_inputs(inputs);
        LOG_IF(ERROR,inputs.size()!=1)<<output.name()<<" has "<<inputs.size()<<" inputs";
        model->add_output(output.name(),inputs[0]);
    }
    for(const auto& initializer : m_model.graph().initializer())
    {
        node_ptr node=model->get_node(initializer.name());
        LOG_IF(FATAL,node==nullptr)<<"Node "<<initializer.name()<<" not found!";
        LOG_IF(FATAL,model->is_edgeEixst(initializer.name())==false)<<"Edge "<<initializer.name()<<" not found!";
        edge_vec outputs=model->get_edge(initializer.name());
        node->set_edge_outputs(outputs);
    }
}

void cel::OnnxParser::build_graph(Model* model) {
    LOG(INFO)<<"start building graph...";
    std::queue<node_ptr> node_que;
    for(auto& input : m_input_info){
        node_ptr node=model->get_node(input.first);
        LOG_IF(FATAL,node==nullptr)<<"Node "<<input.first<<" not found!";
        node_que.push(node);
    }

    while(!node_que.empty()){
        node_ptr cur_node=node_que.front();
        DLOG(INFO)<<cur_node->name()<<" in queue";
        node_que.pop();
        int input_index=0;
        for(auto& edge : cur_node->inputs()){
            node_ptr prev_node=edge->src();
            LOG_IF(FATAL,prev_node==nullptr)<<"Node "<<cur_node->name()<<" has no input node!";
            if(edge->dst()==nullptr){
                edge->set_dst(cur_node);
            }
            if(edge->output_index()!=input_index){
                edge->set_output_index(input_index);
            }
        }
        int output_index=0;
        for(auto& edge : cur_node->outputs()){
            if(edge->src()==nullptr){
                edge->set_src(cur_node);
            }
            if(edge->input_index()!=output_index){
                edge->set_input_index(output_index);
            }
            
            node_ptr next_node=edge->dst();
            LOG_IF(FATAL,next_node==nullptr)<<cur_node->name()<<" has no output node!";
            node_que.push(next_node);
        }
    }

    if(false==model->verify()){
        LOG(ERROR)<<"Model verification failed!";
        return;
    }

    LOG(INFO)<<model->name()<<" graph built successfully!";
    LOG(INFO)<<model->name()<<" has "<<model->get_node_num()<<" nodes and "<<model->get_edge_num()<<" edges";
}

template<typename T>
std::vector<T> cel::OnnxParser::parse_tensor(const onnx::TensorProto& tensor,OnnxType datatype){
    std::vector<T> data;
    if (!tensor.raw_data().empty()) {
        const size_t elem_size = sizeof(T);
        const size_t data_size = tensor.raw_data().size();
        const size_t num_elements = data_size / elem_size;
        LOG_IF(FATAL, data_size % elem_size != 0) << "Raw data size is not a multiple of element size.";
        LOG(INFO)<<"data size:"<<num_elements;
        data.resize(num_elements);
        std::memcpy(data.data(), tensor.raw_data().data(), data_size);
    }else{
        size_t num_elements=0;
        switch (datatype)
        {
        case OnnxType::FLOAT:
            num_elements = tensor.float_data_size();
            data.resize(num_elements);
            std::memcpy(data.data(), tensor.float_data().data(), num_elements * sizeof(float));
            break;
        case OnnxType::INT64:
            num_elements = tensor.int64_data_size();
            data.resize(num_elements);
            std::memcpy(data.data(), tensor.int64_data().data(), num_elements * sizeof(int64_t));
            break;
        case OnnxType::INT32:
            num_elements = tensor.int32_data_size();
            data.resize(num_elements);
            std::memcpy(data.data(), tensor.int32_data().data(), num_elements * sizeof(int32_t));
            break;
        default:
            LOG(WARNING)<<"Data type "<<get_onnx_type(datatype)<<" not supported now!";
            break;
        }
    }
    return data;
}

std::any cel::OnnxParser::parse_attribute(const onnx::AttributeProto &attr) {
    if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT)
    {
        return attr.f();
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT)
    {
        return attr.i();
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING)
    {
        return attr.s();
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS)
    {
        std::vector<float> floats;
        for (const auto& f : attr.floats()) {
            floats.push_back(f);
        }
        return floats;
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS)
    {
        std::vector<int64_t> ints;
        for (const auto& i : attr.ints()) {
            ints.push_back(i);
        }
        return ints;
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR)
    {
        LOG(WARNING)<<"Attribute type TENSOR not supported!";
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS)
    {
        LOG(WARNING)<<"Attribute type TENSORS not supported!";
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH)
    {
        LOG(WARNING)<<"Attribute type GRAPH not supported!";
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPHS)
    {
        LOG(WARNING)<<"Attribute type GRAPHS not supported!";
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSOR)
    {
        LOG(WARNING)<<"Attribute type SPARSE_TENSOR not supported!";
    }else if(attr.type()==onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSORS)
    {
        LOG(WARNING)<<"Attribute type SPARSE_TENSORS not supported!";
    }else{
        LOG(WARNING)<<"Attribute type not supported!";
    }
}

onnx::AttributeProto cel::OnnxParser::to_attribute(const std::string& name,const std::any& value){

}
std::string cel::OnnxParser::get_onnx_type(OnnxType type) {
    switch (type)
    {
    case OnnxType::FLOAT:
        return "FLOAT";
    case OnnxType::UINT8:
        return "UINT8";
    case OnnxType::INT8:
        return "INT8";
    case OnnxType::UINT16:
        return "UINT16";
    case OnnxType::INT16:
        return "INT16";
    case OnnxType::INT32:
        return "INT32";
    case OnnxType::INT64:
        return "INT64";
    case OnnxType::STRING:
        return "STRING";
    case OnnxType::BOOL:
        return "BOOL";
    case OnnxType::FLOAT16:
        return "FLOAT16";
    case OnnxType::DOUBLE:
        return "DOUBLE";
    case OnnxType::UINT32:
        return "UINT32";
    case OnnxType::UINT64:
        return "UINT64";
    case OnnxType::COMPLEX64:
        return "COMPLEX64";
    case OnnxType::COMPLEX128:
        return "COMPLEX128";
    case OnnxType::BFLOAT16:
        return "BFLOAT16";
    default:
        return "UNKNOWN";
    }
}

std::shared_ptr<cel::Node> cel::OnnxParser::create_node(const std::string& node_type)
{
    if(node_type=="Input")
    {
        return std::make_shared<cel::InputNode>();
    }else if(node_type=="Output")
    {
        return std::make_shared<cel::OutputNode>();
    }else if(node_type=="Initializer")
    {
        return std::make_shared<cel::InitializerNode>();
    }else if(node_type=="Conv")
    {
        return std::make_shared<cel::ConvNode>();
    }else if(node_type=="Relu")
    {
        return std::make_shared<cel::ReluNode>();
    }else if(node_type=="MaxPool")
    {
        return std::make_shared<cel::MaxPoolNode>();
    }else if(node_type=="Add")
    {
        return std::make_shared<cel::AddNode>();
    }else if(node_type=="GlobalAveragePool")
    {
        return std::make_shared<cel::GlobalAveragePoolNode>();
    }else if(node_type=="Flatten")
    {
        return std::make_shared<cel::FlattenNode>();
    }else if(node_type=="Gemm")
    {
        return std::make_shared<cel::GemmNode>();
    }else{
        return nullptr;
    }
}
