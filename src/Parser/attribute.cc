#include "Parser/attribute.hpp"

namespace cel{
    void Attribute::insert(const std::string& key,const std::any& value){
        m_attributes[key]=value;
    }

    void Attribute::insert(std::pair<std::string, std::any> &&pair)
    {
        m_attributes[pair.first]=pair.second;
    }

    std::any &Attribute::operator[](const std::string &key)
    {
        if(m_attributes.find(key)==m_attributes.end()){
            m_attributes[key]=std::any();
        }
        return m_attributes[key];
    }

    std::map<std::string, std::any> &Attribute::get_attributes()
    {
        return m_attributes;
    }

    std::any &Attribute::find(const std::string &key)
    {
        if(m_attributes.find(key)==m_attributes.end()){
            m_attributes[key]=std::any();
        }
        return m_attributes[key];
    }
}