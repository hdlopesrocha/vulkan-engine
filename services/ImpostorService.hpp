#pragma once

#include "Service.hpp"
#include "../vulkan/renderer/ImpostorCapture.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>

class VulkanApp;
class VegetationRenderer;

class ImpostorService : public Service {
public:
    ImpostorService();
    void init(VulkanApp* app) override;
    void cleanup() override;

    void setSource(VkImageView albedo, VkImageView normal, VkImageView opacity,
                   VkSampler sampler, int billboardCount);
    void setVegetationRenderer(VegetationRenderer* renderer) { vegRenderer = renderer; }
    void captureAll(float scale);
    void rewire();

    bool isReady() const { return capture.isReady(); }

    VkDescriptorSet getImGuiDescSet(uint32_t billboardType, uint32_t viewIdx) const;
    VkDescriptorSet getImGuiNormalDescSet(uint32_t billboardType, uint32_t viewIdx) const;
    VkDescriptorSet getImGuiDepthDescSet(uint32_t billboardType, uint32_t viewIdx) const;
    const glm::vec3& getViewDir(uint32_t viewIdx) const { return capture.getViewDir(viewIdx); }
    uint32_t closestView(const glm::vec3& dir) const { return capture.closestView(dir); }

private:
    ImpostorCapture capture;
    VulkanApp* vulkanApp = nullptr;
    VegetationRenderer* vegRenderer = nullptr;
    VkImageView srcAlbedo = VK_NULL_HANDLE;
    VkImageView srcNormal = VK_NULL_HANDLE;
    VkImageView srcOpacity = VK_NULL_HANDLE;
    VkSampler srcSampler = VK_NULL_HANDLE;
    int billboardCount = 0;
};
