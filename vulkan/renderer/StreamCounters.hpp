#pragma once

#include <atomic>
#include <cstdint>

// Relaxed atomics written from the build path (worker/main threads).
struct StreamCounters {
    std::atomic<uint64_t> topologyReady{0};
    std::atomic<uint64_t> topologyUpToDate{0};
    std::atomic<uint64_t> topologyUnavailable{0};
    std::atomic<uint64_t> residentBuilt{0};
    std::atomic<uint64_t> residentEmpty{0};
    std::atomic<uint64_t> residentFailed{0};
    std::atomic<uint64_t> cpuFallback{0};
};
