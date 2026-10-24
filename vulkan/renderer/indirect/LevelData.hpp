#pragma once

#include <cstdint>

#include <glm/glm.hpp>

// One chunk mesh's packed span in the shared element pools.
struct LevelData {
    bool allocated = false;
    uint32_t baseVertex = 0;   // absolute element offset (mergedVertices)
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;   // absolute element offset (mergedIndices)
    uint32_t indexCount = 0;
    int level = 0;             // 0-based LoD level (0 = frontier chunk)
    // Ranges of the mesh's PREVIOUS geometry, freed once the replacement
    // upload completes (deferred — in-flight frames may still reference them).
    // UINT32_MAX base = nothing pending.
    uint32_t oldVertexBase = UINT32_MAX;
    uint32_t oldVertexCount = 0;
    uint32_t oldIndexBase = UINT32_MAX;
    uint32_t oldIndexCount = 0;
    glm::vec4 boundsMin = glm::vec4(0.0f);
    glm::vec4 boundsMax = glm::vec4(0.0f);
};
