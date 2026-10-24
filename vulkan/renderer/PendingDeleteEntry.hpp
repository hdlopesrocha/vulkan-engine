#pragma once

#include <cstdint>

// One deferred mesh-slot free, aged out after MAX_FRAMES_IN_FLIGHT.
struct PendingDeleteEntry {
    uint32_t slotIndex = UINT32_MAX;
    uint32_t birthFrame = 0;
};
