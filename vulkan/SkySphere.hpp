#pragma once

#include "VulkanApp.hpp"
#include <vector>
#include <memory>

struct SkySettings;

class SkySphere {
public:
    explicit SkySphere();
    ~SkySphere();

    // Initialize sky buffer and bind into provided descriptor sets (binding 6)
    void init(VulkanApp* app, SkySettings& settings,
              VkDescriptorSet descriptorSet);

    // Update sky UBO contents from SkySettings (call per-frame if UI may change)
    void update(VulkanApp* app);

    // Destroy GPU resources
    void cleanup();

private:
    Buffer skyBuffer{};
    VkDeviceSize skyBufferSize = 0;
    SkySettings* skySettings = nullptr;
    // Note: no stored VulkanApp*; callers must pass VulkanApp* to init/update as needed
};
