#pragma once

// Active brush control mode (shared across all controllers).
enum class BrushControlMode {
    TRANSLATE,
    AIM,
    SCALE,
    TEXTURE,
    ATTRIBUTE,
    COLOR,
    MODE_COUNT
};
