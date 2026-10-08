#pragma once

// One rock anchor: world position + per-instance variation. Produced by the
// rock collector (brush-7 triangles); the SDF scene itself stays
// independent of terrain/octree representations. CPU-only input to
// createRocksFromAnchors (becomes a Rock Instance).
#include <glm/glm.hpp>

namespace sdf_gpu {

struct RockAnchor {
    glm::vec3 pos = glm::vec3(0.0f);
    glm::vec3 euler = glm::vec3(0.0f); // R = Rx * Ry * Rz (random tumble)
    float scale = 1.0f;                // world radius (m)
    float seed = 0.0f;
    float intensity = 1.0f;
};

} // namespace sdf_gpu
