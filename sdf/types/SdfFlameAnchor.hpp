#pragma once

// One flame anchor: world position + per-instance variation. Produced by
// the lava collector (brush-4 triangles) or any other emitter; the SDF
// scene itself stays independent of terrain/octree representations.
// CPU-only input to createFireFromAnchors (becomes a Flame Instance).
#include <glm/glm.hpp>

namespace sdf_gpu {

struct FlameAnchor {
    glm::vec3 pos = glm::vec3(0.0f);
    glm::vec3 euler = glm::vec3(0.0f); // R = Rx * Ry * Rz, local +Y = flame axis
    float scale = 1.0f;
    float heightScale = 1.0f; // per-instance vertical stretch
    float seed = 0.0f;
    float intensity = 1.0f;
};

} // namespace sdf_gpu
