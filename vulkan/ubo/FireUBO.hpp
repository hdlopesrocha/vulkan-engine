#pragma once
#include <glm/glm.hpp>

// GPU uniform buffer for animated fire billboard parameters.
// Accessed at set=2, binding=1 in vegetation shaders (vert + frag + depth).
// Updated once per frame (or less frequently for invariant fields).
// std140: 7 x vec4 = 112 bytes.
struct FireParamsUBO {
    // x = enabled (1/0), y = size (world units, width), z = speed, w = intensity
    glm::vec4 enabledSizeSpeedIntensity = glm::vec4(1.0f, 6.0f, 2.0f, 1.2f);
    // x = flicker (0..1), y = noiseScale, z = heightScale, w = turbulence
    glm::vec4 shape = glm::vec4(0.5f, 2.5f, 1.6f, 0.6f);
    // x = riseSpeed, y = windInfluence, z = alpha, w = emissive
    glm::vec4 motion = glm::vec4(1.5f, 0.35f, 1.0f, 1.2f);
    // rgb = color, w unused
    glm::vec4 innerColor = glm::vec4(1.0f, 0.95f, 0.60f, 0.0f);
    glm::vec4 midColor   = glm::vec4(1.0f, 0.45f, 0.10f, 0.0f);
    glm::vec4 outerColor = glm::vec4(0.60f, 0.05f, 0.00f, 0.0f);
    // x = smoke amount (0..1), yzw spare
    glm::vec4 extra = glm::vec4(0.25f, 0.0f, 0.0f, 0.0f);
};
static_assert(sizeof(FireParamsUBO) == 112, "FireParamsUBO expected 112 bytes");
