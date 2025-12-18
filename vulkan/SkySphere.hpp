#pragma once

#include "VulkanApp.hpp"
#include <vector>
#include <memory>

class SkyWidget;

class SkySphere {
public:
    explicit SkySphere(VulkanApp* app);
    ~SkySphere();

    // Initialize sky buffer and bind into provided descriptor sets (binding 6)
    void init(SkyWidget* widget,
              std::vector<VkDescriptorSet>& descriptorSets,
              VkDescriptorSet shadowDescriptorSet);

    // Update sky UBO contents from widget (call per-frame if UI may change)
    void update();

    // Destroy GPU resources
    void cleanup();

private:
    VulkanApp* app = nullptr;
    Buffer skyBuffer{};
    VkDeviceSize skyBufferSize = 0;
    SkyWidget* skyWidget = nullptr;
};
