#pragma once

#include <atomic>
#include <cstdint>

// Render-loop vkUpdateDescriptorSets accounting.
//
// Validation for the descriptor-buffer migration: the render loop must issue
// zero vkUpdateDescriptorSets calls in steady state (all static bindings are
// written once at init / on resource change; per-frame UBO contents are
// streamed via memcpy / vkCmdCopyBuffer into the already-bound buffers).
// Every DescriptorWriter::flush() records its write count here so a CPU
// profile (or a debug snapshot around drawFrame) can assert the delta is 0.
struct DescriptorUpdateStats {
    static inline std::atomic<uint64_t> s_updateCalls{0};
    static void noteUpdate(uint64_t n = 1) {
        s_updateCalls.fetch_add(n, std::memory_order_relaxed);
    }
    static void reset() { s_updateCalls.store(0, std::memory_order_relaxed); }
};
