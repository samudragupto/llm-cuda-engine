#pragma once
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>

struct Profiler {
    struct EventPair { cudaEvent_t start, stop; float accumulated_ms; int count; };
    std::unordered_map<std::string, EventPair> events;
    std::vector<std::string> order;

    void start(const std::string& name) {
        if (events.find(name) == events.end()) {
            cudaEventCreate(&events[name].start);
            cudaEventCreate(&events[name].stop);
            events[name].accumulated_ms = 0.0f;
            events[name].count = 0;
            order.push_back(name);
        }
        cudaEventRecord(events[name].start);
    }

    void stop(const std::string& name) {
        cudaEventRecord(events[name].stop);
        cudaEventSynchronize(events[name].stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, events[name].start, events[name].stop);
        events[name].accumulated_ms += ms;
        events[name].count++;
    }

    void print_summary() {
        printf("\n=== KERNEL PROFILING SUMMARY ===\n");
        printf("%-25s | %-10s | %-12s | %-10s\n", "Kernel / Event", "Calls", "Total (ms)", "Avg (ms)");
        printf("--------------------------|------------|--------------|------------\n");
        float total_all = 0.0f;
        for (const auto& name : order) {
            auto& ev = events[name];
            float avg = ev.count > 0 ? ev.accumulated_ms / ev.count : 0.0f;
            total_all += ev.accumulated_ms;
            printf("%-25s | %-10d | %-12.4f | %-10.4f\n", name.c_str(), ev.count, ev.accumulated_ms, avg);
        }
        printf("--------------------------|------------|--------------|------------\n");
        printf("%-25s | %-10s | %-12.4f |\n\n", "TOTAL RECORDED TIME", "-", total_all);
    }
};