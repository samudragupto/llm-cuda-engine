#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "tensor.h"
#include "mem_pool.h"

struct SafetensorsLoader {
    struct Entry {
        std::string dtype;
        std::vector<int> shape;
        size_t offset_start, offset_end;
    };
    std::unordered_map<std::string, Entry> entries;
    std::string filepath;
    uint64_t header_size;
    std::vector<char> raw;

    bool load(const std::string& path) {
        filepath = path;
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.read((char*)&header_size, 8);
        std::string header(header_size, '\0');
        f.read(&header[0], header_size);
        f.seekg(0, std::ios::end);
        size_t total = (size_t)f.tellg();
        size_t data_size = total - 8 - (size_t)header_size;
        raw.resize(data_size);
        f.seekg(8 + (std::streamoff)header_size);
        f.read(raw.data(), data_size);
        parse_header(header);
        return true;
    }

    void parse_header(const std::string& h) {
        size_t pos = 0;
        while ((pos = h.find("\"", pos)) != std::string::npos) {
            size_t ks = pos + 1;
            size_t ke = h.find("\"", ks);
            if (ke == std::string::npos) break;
            std::string key = h.substr(ks, ke - ks);
            pos = ke + 1;
            if (key == "__metadata__") {
                pos = h.find("}", pos);
                if (pos != std::string::npos) pos++;
                continue;
            }
            Entry e;
            size_t ds = h.find("\"dtype\"", pos);
            if (ds == std::string::npos) break;
            size_t dvs = h.find("\"", ds + 7);
            if (dvs == std::string::npos) break;
            dvs++;
            size_t dve = h.find("\"", dvs);
            if (dve == std::string::npos) break;
            e.dtype = h.substr(dvs, dve - dvs);

            size_t shape_key = h.find("\"shape\"", pos);
            if (shape_key == std::string::npos) break;
            size_t ss = h.find("[", shape_key);
            size_t se = h.find("]", ss);
            if (ss == std::string::npos || se == std::string::npos) break;
            std::string shapes = h.substr(ss + 1, se - ss - 1);
            size_t sp = 0;
            while (sp < shapes.size()) {
                size_t np = shapes.find(",", sp);
                if (np == std::string::npos) np = shapes.size();
                std::string num = shapes.substr(sp, np - sp);
                while (!num.empty() && (num[0] == ' ' || num[0] == '\n')) num.erase(0, 1);
                while (!num.empty() && (num.back() == ' ' || num.back() == '\n')) num.pop_back();
                if (!num.empty()) e.shape.push_back(std::stoi(num));
                sp = np + 1;
            }

            size_t off_key = h.find("\"data_offsets\"", pos);
            if (off_key == std::string::npos) break;
            size_t os = h.find("[", off_key);
            size_t oe = h.find("]", os);
            if (os == std::string::npos || oe == std::string::npos) break;
            std::string offsets = h.substr(os + 1, oe - os - 1);
            size_t comma = offsets.find(",");
            if (comma == std::string::npos) break;
            std::string s1 = offsets.substr(0, comma);
            std::string s2 = offsets.substr(comma + 1);
            while (!s1.empty() && s1[0] == ' ') s1.erase(0, 1);
            while (!s2.empty() && s2[0] == ' ') s2.erase(0, 1);
            e.offset_start = (size_t)std::stoull(s1);
            e.offset_end = (size_t)std::stoull(s2);

            entries[key] = e;
            pos = h.find("}", pos);
            if (pos != std::string::npos) pos++;
        }
    }

    
    bool validate_shape(const std::string& name, const std::vector<int>& expected) {
        auto it = entries.find(name);
        if (it == entries.end()) return false; 
        const auto& actual = it->second.shape;
        if (actual.size() != expected.size()) return false;
        for (size_t i = 0; i < actual.size(); i++) {
            if (actual[i] != expected[i]) return false;
        }
        return true;
    }

    Tensor load_tensor(const std::string& name, MemPool& pool) {
        auto it = entries.find(name);
        if (it == entries.end()) {
            fprintf(stderr, "[FATAL] Tensor '%s' not found in safetensors.\n", name.c_str());
            exit(1);
        }
        Entry& e = it->second;
        Tensor t(pool, e.shape);
        size_t bytes = e.offset_end - e.offset_start;

        if (e.dtype == "F32") {
            CUDA_CHECK(cudaMemcpy(t.data, raw.data() + e.offset_start, bytes, cudaMemcpyHostToDevice));
        } else {
            fprintf(stderr, "[FATAL] Unsupported dtype '%s' for tensor '%s'. Require F32.\n", e.dtype.c_str(), name.c_str());
            exit(1);
        }
        return t;
    }

    void list_tensors() {
        for (auto& kv : entries) {
            printf("  %-55s dtype=%-4s shape=[", kv.first.c_str(), kv.second.dtype.c_str());
            for (size_t i = 0; i < kv.second.shape.size(); i++) {
                printf("%d", kv.second.shape[i]);
                if (i < kv.second.shape.size() - 1) printf(",");
            }
            printf("] bytes=%zu\n", kv.second.offset_end - kv.second.offset_start);
        }
    }
};