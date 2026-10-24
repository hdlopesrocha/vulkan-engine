#pragma once

#include <cstdint>

struct MeshInfo;

// A reserved span plus the MeshInfo entry it will be published into.
struct SpanReservation {
    uint32_t slotIndex = UINT32_MAX;
    MeshInfo* info = nullptr;
    uint32_t baseVertex = 0;
    uint32_t firstIndex = 0;
};
