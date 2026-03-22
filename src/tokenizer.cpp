#include "tokenizer.h"
#include <cstdio>
#include <cstdlib>

void LlamaTokenizer::load(const char* path) { 
    FILE* f = fopen(path, "rb"); 
    if (!f) { printf("[FATAL] Could not open %s\n", path); exit(1); }
    int vs; fread(&vs, 4, 1, f); 
    vocab.resize(vs, ""); 
    for (int i = 0; i < vs; i++) { 
        int l = 0; fread(&l, 4, 1, f); 
        if (l > 0 && l < 1000) {
            std::string s(l, '\0'); fread(&s[0], 1, l, f); vocab[i] = s; 
        }
    } 
    fclose(f); 
}
std::string LlamaTokenizer::decode(int id) { 
    if (id <= 2 || id >= vocab.size()) return ""; 
    return vocab[id]; 
}
std::string LlamaTokenizer::decode(const std::vector<int>& ids) { 
    std::string s; for (int id : ids) s += decode(id); return s; 
}