#pragma once

#include "VulkanApp.hpp"
#include <vector>
#include <memory>

struct SkySettings;

class SkySphere {
public:
    explicit SkySphere(VulkanApp* app);
    ~SkySphere();

    // Initialize sky buffer and bind into provided descriptor sets (binding 6)
    void init(SkySettings& settings,
              VkDescriptorSet descriptorSet);

    // Update sky UBO contents from SkySettings (call per-frame if UI may change)
    void update();

    // Destroy GPU resources
    void cleanup();

private:
    VulkanApp* app = nullptr;
    Buffer skyBuffer{};
    VkDeviceSize skyBufferSize = 0;
    SkySettings* skySettings = nullptr;
};
