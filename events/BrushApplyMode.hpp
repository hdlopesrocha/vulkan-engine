#pragma once

#include <cstdint>

// How the brush apply key behaves for a given controller:
//   Click — apply once per key press (edge-triggered)
//   Drag  — apply continuously every frame while held (default)
enum class BrushApplyMode : uint8_t {
    Click,
    Drag
};
