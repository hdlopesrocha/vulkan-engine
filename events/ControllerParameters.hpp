#pragma once

#include <glm/glm.hpp>
#include <cstdint>

// How the brush apply key behaves for a given controller:
//   Click — apply once per key press (edge-triggered)
//   Drag  — apply continuously every frame while held (default)
enum class BrushApplyMode : uint8_t {
    Click,
    Drag
};

// Shared speed/tunable parameters for all controllers. Page selection now lives
// in ControllerContext (a tree of pages per controller), so this struct only
// carries numeric tunables and per-controller brush apply modes.
class ControllerParameters {
public:
    ControllerParameters() = default;

    // Camera-related parameters
    float cameraMoveSpeed = 8.0f;
    float cameraAngularSpeedDeg = 90.0f;
    float nunchukTransSpeed = 256.0f;
    float nunchukRotSpeed = 90.0f;
    float wiimoteTransSpeed = 256.0f;
    float wiimoteRotSpeed = 90.0f;

    // Per-controller brush apply mode (Click vs Drag). All default to Drag.
    BrushApplyMode keyboardBrushMode = BrushApplyMode::Drag;
    BrushApplyMode mouseBrushMode    = BrushApplyMode::Drag;
    BrushApplyMode gamepadBrushMode  = BrushApplyMode::Drag;
    BrushApplyMode wiimoteBrushMode  = BrushApplyMode::Drag;
};
