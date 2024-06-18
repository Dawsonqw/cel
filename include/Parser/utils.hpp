#ifndef BASE_UTILS_H
#define BASE_UTILS_H
#include <memory>
#include <vector>

namespace cel{
    class Node;
    class Edge;

    using edge_ptr=std::shared_ptr<Edge>;
    using edge_vec=std::vector<edge_ptr>;
    using node_ptr=std::shared_ptr<Node>;
}
#endif