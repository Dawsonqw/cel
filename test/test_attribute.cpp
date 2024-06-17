#include <gtest/gtest.h>
#include "Parser/attribute.hpp"

TEST(AttributeTest, test_attribute){
    cel::Attribute attr;
    attr.insert("int",1);
    std::any& int_value=attr.find("int");
    int val_i=std::any_cast<int>(int_value);
    EXPECT_EQ(val_i,1);

    attr.insert("string","value");
    std::any& string_value=attr.find("string");
    const char* val_s=std::any_cast<const char*>(string_value);
    EXPECT_STREQ(val_s,"value");

    attr.insert("double",3.14);
    std::any& double_value=attr.find("double");
    double val_d=std::any_cast<double>(double_value);
    EXPECT_DOUBLE_EQ(val_d,3.14);

    std::any& _value=attr["int"];
    int val_=std::any_cast<int>(int_value);
    EXPECT_EQ(val_,1);
}