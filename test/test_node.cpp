#include <gtest/gtest.h>
#include "Parser/node.hpp"
#include "Parser/edge.hpp"
#include "Parser/attribute.hpp"

TEST(AttributeTest, test_attribute){
    cel::Attribute attr;
    attr.insert("type","int");
    attr.insert({"value",1});
    std::string node_name="node_test";
    std::string node_type="node_type";
    cel::Edge edge("edge_test");
    cel::Node node(node_name,node_type,attr);

    ASSERT_EQ(node.name(),node_name);
    node.set_name("node_test2");
    ASSERT_EQ(node.name(),"node_test2");

}