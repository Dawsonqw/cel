#ifndef BASE_PARSER_HPP
#define BASE_PARSER_HPP
#include <string>

namespace cel{
    class Parser{
        public:
            Parser(const std::string& file_name):m_filename(file_name){}
            virtual ~Parser()=default;

            Parser(const Parser&)=delete;
            Parser& operator=(const Parser&)=delete;

            virtual void parse()=0;
        protected:
            const std::string m_filename;
    };
}
#endif