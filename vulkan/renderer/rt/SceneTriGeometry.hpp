#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

#include <glm/glm.hpp>

// One real chunk triangle range fed to the scene BLAS.
struct SceneTriGeometry {
    VkDeviceAddress vertexAddress = 0; // chunk's first vertex (already offset)
    VkDeviceAddress indexAddress = 0;  // chunk's first index (already offset)
    // Owning merged buffers (solid or water pools). The async AS build
    // records its own TRANSFER -> ACCEL_BUILD barrier for these (the
    // uploads that fill them are vkCmdCopyBuffer submissions), so the
    // build no longer depends on the cull CB's acquireBuffers barrier.
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    uint32_t baseVertex = 0;  // element offset into the merged vertex pool
    uint32_t firstIndex = 0;  // element offset into the merged index pool
    glm::vec4 albedo = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
};
