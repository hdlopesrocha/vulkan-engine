#pragma once

#include "../core/VulkanApp.hpp"
#include <vector>
#include <memory>

struct SkySettings;
struct CloudSettings;

class SkySphere {
public:
    explicit SkySphere();
    ~SkySphere();

    // Initialize sky buffer and bind into provided descriptor sets (binding 6)
    void init(VulkanApp* app, SkySettings& settings,
              VkDescriptorSet descriptorSet);
    // Attach cloud settings + animation clock (clouds share the SkyUBO).
    void setCloudSettings(CloudSettings* clouds) { cloudSettings = clouds; }
    void setCloudTime(float t) { cloudTime = t; }
    void setCloudsEnabled(bool e) { cloudsEnabledOverride = e; hasCloudsOverride = true; }
    // Cloud ray-cast quality (Settings::raycastPixelSize): one cloud ray per
    // NxN screen-pixel block (block center). 1 = per pixel.
    void setRaycastPixelSize(int px) { raycastPixelSize_ = (px < 1) ? 1 : ((px > 8) ? 8 : px); }

    // Write sky UBO buffer to an additional descriptor set (for multi-frame setups)
    void writeDescriptorSet(VulkanApp* app, VkDescriptorSet descriptorSet);

    // Update sky UBO contents from SkySettings (call per-frame if UI may change)
    void update(VulkanApp* app);

    // Access the sky uniform buffer for binding to descriptor sets
    Buffer getBuffer() const { return skyBuffer; }

    // Destroy GPU resources
    void cleanup();

private:
    void fillSkyUniform(struct SkyUniform& out) const;
    Buffer skyBuffer{};
    VkDeviceSize skyBufferSize = 0;
    SkySettings* skySettings = nullptr;
    CloudSettings* cloudSettings = nullptr;
    float cloudTime = 0.0f;
    bool cloudsEnabledOverride = true;
    bool hasCloudsOverride = false;
    // Settings::raycastPixelSize mirror (cloud march block size).
    int raycastPixelSize_ = 2;
    // Note: no stored VulkanApp*; callers must pass VulkanApp* to init/update as needed
};
