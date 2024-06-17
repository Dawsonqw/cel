#include <iostream>
#include <variant>
#include <map>
#include <vector>
#include <concepts>
#include <any>
template<typename T>
concept AttributeType=std::same_as<T,std::string>||std::same_as<T,int>||
                std::same_as<T,std::vector<int>>||std::same_as<T,std::vector<float>>||
                std::same_as<T,std::vector<std::string>>||std::same_as<T,std::map<std::string,std::string>>;

int main(){
    // using value_type=std::variant<int,float,std::string>;
    // std::map<std::string,value_type> m;
    // m.insert({"int",1});
    // m.insert({"float",1.0});
    // m.insert({"string","hello"});

    // 使用 any
std::map<std::string, std::any> m;
    m.insert({"int", 1});
    m.insert({"float", 1.0});
    m.insert({"string", "hello"});
    m.insert({"vector_int", std::vector<int>{1, 2, 3}});

    for (const auto& [key, val] : m) {
        std::cout << key << ": ";
        try {
            if (val.type() == typeid(int)) {
                std::cout << std::any_cast<int>(val) << std::endl;
            } else if (val.type() == typeid(double)) {
                std::cout << std::any_cast<double>(val) << std::endl;
            } else if (val.type() == typeid(const char*)) {
                std::cout << std::any_cast<const char*>(val) << std::endl;
            } else if (val.type() == typeid(std::vector<int>)) {
                for (const auto& i : std::any_cast<std::vector<int>>(val)) {
                    std::cout << i << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "Unsupported type: " << val.type().name() << std::endl;
            }
        } catch (const std::bad_any_cast& e) {
            std::cout << "Failed to cast due to bad_any_cast exception." << std::endl;
        }
    }
    return 0;
}