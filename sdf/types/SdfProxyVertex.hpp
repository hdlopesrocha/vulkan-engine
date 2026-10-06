#pragma once

// Unit proxy-cube vertex ([0,1]^3 positions, mapped onto container AABBs).
// CPU-side Vulkan vertex (bound by attributes, no GLSL struct).
#include <glm/glm.hpp>

struct SdfProxyVertex {
    glm::vec3 position;
};
