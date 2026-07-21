#pragma once

#include <glm/glm.hpp>

// Shared speed/tunable parameters for all controllers. Page selection now lives
// in ControllerContext (a tree of pages per controller), so this struct only
// carries numeric tunables.
class ControllerParameters {
public:
    ControllerParameters() = default;

    // Camera-related parameters
    float cameraMoveSpeed = 8.0f;
    float cameraAngularSpeedDeg = 90.0f;
    float nunchukTransSpeed = 256.0f;
    float nunchukRotSpeed = 90.0f;
};
