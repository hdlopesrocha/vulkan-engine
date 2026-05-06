#pragma once
#include <vulkan/vulkan.h>

class VulkanApp;

class CubeToEquirectRenderer {
public:
    CubeToEquirectRenderer();
    ~CubeToEquirectRenderer();

    void cleanup(VulkanApp* app);
    void render(VulkanApp* app, VkSampler sampler, VkImageView cubeMapView);
    VkImageView getEquirectView() const { return cube360EquirectView; }
    VkDescriptorSet getDescriptor() const { return cube360EquirectSampleDescriptorSet; }
private:
    void ensureResources(VulkanApp* app);
    void createPipeline(VulkanApp* app);
    void createOutputTarget(VulkanApp* app);
    void createDescriptorResources(VulkanApp* app, VkSampler sampler, VkImageView cubeMapView);

    VkImage cube360EquirectImage = VK_NULL_HANDLE;
    VkDeviceMemory cube360EquirectMemory = VK_NULL_HANDLE;
    VkImageView cube360EquirectView = VK_NULL_HANDLE;
    VkPipeline cube360EquirectPipeline = VK_NULL_HANDLE;
    VkPipelineLayout cube360EquirectPipelineLayout = VK_NULL_HANDLE;
    VkShaderModule cube360EquirectVertModule = VK_NULL_HANDLE;
    VkShaderModule cube360EquirectFragModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout cube360EquirectDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet cube360EquirectSampleDescriptorSet = VK_NULL_HANDLE;

    static constexpr uint32_t EQ_WIDTH = 1024;
    static constexpr uint32_t EQ_HEIGHT = 512;
};
