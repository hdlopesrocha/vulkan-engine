#pragma once

// SDF instance: placement of one shared definition (transform + palette
// indices + animation seed). Definitions are never duplicated per instance.
#include <glm/glm.hpp>
#include <cstdint>

namespace sdf_gpu {

struct Instance {
    uint32_t defIdx = 0;
    uint32_t matIdx = 0;
    glm::vec3 pos = glm::vec3(0.0f);
    glm::vec3 euler = glm::vec3(0.0f);
    float scale = 1.0f;
    float heightScale = 1.0f;
    float radiusScale = 1.0f;
    float intensity = 1.0f;
    float seed = 0.0f;
    uint32_t containerIdx = 0;
};

} // namespace sdf_gpu
