#pragma once
#include "Renderer.hpp"
#include <vulkan/vulkan.h>
#include "../VmaContext.hpp"
#include "../TrackedHandle.hpp"
#include "CommandBufferState.hpp"

class VulkanApp;

class CubeToEquirectRenderer : public Renderer {
public:
    CubeToEquirectRenderer();
    ~CubeToEquirectRenderer();

    void cleanup(VulkanApp* app) override;
    void render(VulkanApp* app, VkSampler sampler, VkImageView cubeMapView);
    VkImageView getEquirectView() const { return cube360EquirectView; }
    VkDescriptorSet getDescriptor() const { return cube360EquirectSampleDescriptorSet; }
private:
    void ensureResources(VulkanApp* app);
    void createPipeline(VulkanApp* app);
    void createOutputTarget(VulkanApp* app);
    void createDescriptorResources(VulkanApp* app, VkSampler sampler, VkImageView cubeMapView);

    VkImage cube360EquirectImage = VK_NULL_HANDLE;
    VmaAllocation cube360EquirectAllocation = VK_NULL_HANDLE;
    VkDeviceMemory cube360EquirectMemory = VK_NULL_HANDLE;
    VkImageView cube360EquirectView = VK_NULL_HANDLE;
    TrackedHandle<VkPipeline> cube360EquirectPipeline;
    TrackedHandle<VkPipelineLayout> cube360EquirectPipelineLayout;
    TrackedHandle<VkShaderModule> cube360EquirectVertModule;
    TrackedHandle<VkShaderModule> cube360EquirectFragModule;
    TrackedHandle<VkDescriptorSetLayout> cube360EquirectDescriptorSetLayout;
    TrackedHandle<VkDescriptorSet> cube360EquirectSampleDescriptorSet;
    // Cache of the last-written (sampler, view) pair; skip the descriptor flush when unchanged
    VkSampler lastSampler = VK_NULL_HANDLE;
    VkImageView lastCubeMapView = VK_NULL_HANDLE;

    static constexpr uint32_t EQ_WIDTH = 1024;
    static constexpr uint32_t EQ_HEIGHT = 512;
};
