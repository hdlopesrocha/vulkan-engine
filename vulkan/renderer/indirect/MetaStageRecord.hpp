#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

#include <glm/glm.hpp>

// One staged draw-command/meta write awaiting flush.
struct MetaStageRecord {
    uint32_t entryIndex = 0;
    VkDrawIndexedIndirectCommand cmd{};
    glm::vec4 bounds[4] = { glm::vec4(0.0f), glm::vec4(0.0f), glm::vec4(0.0f), glm::vec4(0.0f) };
    bool boundsValid = false;
};
