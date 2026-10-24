#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

// Snapshot of every scene-static descriptor source. Compared by
// updateTextureDescriptorSet() so redundant allocation-listener callbacks are
// a no-op.
struct StaticTextureSignature {
    VkSampler samplers[5] = {};
    VkImageView views[5] = {};
    VkSampler shadowSampler = VK_NULL_HANDLE;
    VkImageView shadowViews[3] = {};
    VkBuffer materials = VK_NULL_HANDLE;
    VkBuffer waterParams = VK_NULL_HANDLE;
    VkBuffer brushMap = VK_NULL_HANDLE;
    // Bumped by TextureArrayManager::allocate()/destroy(). Included so a
    // (re)allocation always forces the descriptor refresh even if the
    // driver recycles image/sampler handles to the previous values
    // (perf report 23 C1 follow-up).
    uint32_t textureVersion = 0;
    bool valid = false;
    bool matches(const StaticTextureSignature& o) const {
        for (int i = 0; i < 5; ++i)
            if (samplers[i] != o.samplers[i] || views[i] != o.views[i]) return false;
        if (shadowSampler != o.shadowSampler) return false;
        for (int i = 0; i < 3; ++i)
            if (shadowViews[i] != o.shadowViews[i]) return false;
        return materials == o.materials && waterParams == o.waterParams &&
               brushMap == o.brushMap &&
               textureVersion == o.textureVersion;
    }
};
