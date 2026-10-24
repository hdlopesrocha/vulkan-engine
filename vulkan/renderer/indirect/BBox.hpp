#pragma once

#include <glm/glm.hpp>

// One debug bounding box submitted to the bbox cull.
struct BBox {
    glm::vec3 minp;
    glm::vec3 maxp;
    // LoD meta so the bbox cull can apply the SAME clipmap band gate as the
    // solid terrain (keeps exactly one rung per region, no overlap). Mirrors
    // the solid bounds entry: cellSize = chunk cube side, level = chunkLod
    // rung, base = chunk min corner (shared column anchor).
    float cellSize = 0.0f;
    int level = 0;
    glm::vec3 base = glm::vec3(0.0f);
};
