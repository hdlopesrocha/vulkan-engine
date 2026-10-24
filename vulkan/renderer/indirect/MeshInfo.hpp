#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

#include <glm/glm.hpp>

#include "LevelData.hpp"

// One registered chunk mesh (draw metadata + packed-pool span).
struct MeshInfo {
    uint32_t id = UINT32_MAX;
    uint32_t baseVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    glm::vec4 boundsMin = glm::vec4(0.0f); // object-space AABB min (xyz)
    glm::vec4 boundsMax = glm::vec4(0.0f); // object-space AABB max (xyz)
    VkDeviceSize indirectOffset = 0; // byte offset into indirect buffer
    uint32_t drawIndex = UINT32_MAX; // position in indirectCommands list
    uint32_t slotIndex = UINT32_MAX; // stable slot (if using slotted mode)
    bool active = false;
    // Emitting cell's min corner (published as the bounds entry's fourth
    // vec4). The hierarchical LoD gate derives parent cells from the tree
    // root lattice, not from this field, so it is informational only now.
    glm::vec4 boundsBase = glm::vec4(0.0f);
    // Single chunk mesh (slotted mode): one packed span in the shared
    // vertex/index element pools, one draw entry per chunk. Chunks arrive
    // one by one — each chunk publishes exactly one mesh.
    LevelData level_ = {};
    // NOTE: per-mesh buffers removed — meshes are packed into the merged buffers
};
