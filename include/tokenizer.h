#pragma once
#include <string>
#include <vector>

struct LlamaTokenizer { 
    std::vector<std::string> vocab; 
    void load(const char* path); 
    std::string decode(int id); 
    std::string decode(const std::vector<int>& ids); 
};