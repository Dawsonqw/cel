#include "Parser/attribute.hpp"

namespace cel{
    void Attribute::insert(const std::string& key,const std::any& value){
        m_attributes[key]=value;
    }

    std::any& Attribute::operator[](const std::string& key){
        if(m_attributes.find(key)==m_attributes.end()){
            m_attributes[key]=std::any();
        }
        return m_attributes[key];
    }

    std::any& Attribute::find(const std::string& key){
        if(m_attributes.find(key)==m_attributes.end()){
            m_attributes[key]=std::any();
        }
        return m_attributes[key];
    }

}