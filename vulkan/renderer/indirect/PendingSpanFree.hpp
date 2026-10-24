#pragma once

#include <cstdint>

// A replaced packed-pool span waiting for the next staged meta flush.
struct PendingSpanFree {
    uint32_t baseVertex = UINT32_MAX;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = UINT32_MAX;
    uint32_t indexCount = 0;
};
