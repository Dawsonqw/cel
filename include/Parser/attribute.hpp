#ifndef BASE_ATTRIBUTE_H
#define BASE_ATTRIBUTE_H
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <any>

namespace cel{
    class Attribute{
        public:
        public:
            Attribute()=default;
            virtual ~Attribute()=default;

            Attribute(const Attribute&)=delete;
            Attribute& operator=(const Attribute&)=delete;

            void insert(const std::string& key,const std::any& value);

            std::any& find(const std::string& key);

            std::any& operator[](const std::string& key);
        
        private:
            std::map<std::string,std::any> m_attributes;
    };
}

#endif