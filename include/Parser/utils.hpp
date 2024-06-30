#ifndef BASE_UTILS_H
#define BASE_UTILS_H
#include <memory>
#include <vector>
#include "tensor.hpp"

namespace cel{
    class Node;
    class Edge;

    using edge_ptr=std::shared_ptr<Edge>;
    using edge_vec=std::vector<edge_ptr>;
    using node_ptr=std::shared_ptr<Node>;

    using node_map_t=std::map<std::string,node_ptr>;
    using edge_map_t=std::map<std::string,edge_ptr>;
    using edge_map=std::map<std::string,edge_vec>;

    template<typename T>
    using tensor_vec=std::vector<std::shared_ptr<Tensor<T>>>;
}
#endif