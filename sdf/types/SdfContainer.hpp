#pragma once

// SDF container: a broadphase volume (world AABB + uniform-grid resolution)
// holding the instances whose bounds overlap it.
#include "math/BoundingBox.hpp"
#include <glm/glm.hpp>
#include <cstdint>
#include <vector>

namespace sdf_gpu {

struct Container {
    BoundingBox aabb = BoundingBox(glm::vec3(0.0f), glm::vec3(1.0f));
    glm::uvec3 resolution = glm::uvec3(8u, 8u, 8u);
    std::vector<uint32_t> instanceIndices;
};

} // namespace sdf_gpu
