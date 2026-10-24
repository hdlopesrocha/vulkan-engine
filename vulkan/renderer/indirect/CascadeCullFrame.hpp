#pragma once

#include <array>

#include "../../resources/Buffer.hpp"

// Per-frame cascade cull outputs (one compact/count buffer per cascade).
struct CascadeCullFrame {
    std::array<Buffer, 3> compactBuffers; // compact output per cascade (DEVICE_LOCAL)
    std::array<Buffer, 3> countBuffers;   // visible count per cascade (DEVICE_LOCAL)
};
