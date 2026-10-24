#pragma once

#include <cstdint>

// Chunk-streaming counters (diagnostic; VULKAN_STREAM_DIAG prints them):
// how the GPU topology path resolved every node it was asked to build.
struct StreamStats {
    uint64_t topologyReady = 0;
    uint64_t topologyUpToDate = 0;
    uint64_t topologyUnavailable = 0;
    uint64_t residentBuilt = 0;   // resident pass produced geometry
    uint64_t residentEmpty = 0;   // resident pass ran, no triangles
    uint64_t residentFailed = 0;  // span/sizing failure (chunk skipped)
    uint64_t cpuFallback = 0;     // resident disabled/unavailable -> CPU walk
};
