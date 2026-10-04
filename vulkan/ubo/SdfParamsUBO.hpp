#pragma once

// Global raymarch / debug parameters (one small UBO, std430, 64 bytes).
#include <glm/glm.hpp>
#include <cstdint>

struct SdfParamsUBO {
    glm::vec4 timeDebug{0.0f};   // x = globalTime, y = debugMode (float), z = maxSteps, w = safetyFactor
    glm::vec4 marchParams{0.0f}; // x = minStep, y = maxStep, z = hitEpsilon, w = earlyTermThreshold
    glm::vec4 fireColors0{0.0f}; // emission gradient low (kept generic)
    glm::vec4 fireColors1{0.0f}; // emission gradient high (kept generic)
};
static_assert(sizeof(SdfParamsUBO) == 64, "SdfParamsUBO must be 64 bytes");
static_assert(sizeof(SdfParamsUBO) % 16 == 0, "SdfParamsUBO must be multiple of 16");
