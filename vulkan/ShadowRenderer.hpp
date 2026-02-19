#pragma once

#include "vulkan.hpp"

class ShadowRenderer {
public:
    ShadowRenderer(uint32_t shadowMapSize = 2048);
    ~ShadowRenderer();
    void init(VulkanApp* app);
    void cleanup(VulkanApp* app);
    // Render shadow pass for a collection of objects
    void beginShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, const glm::mat4& lightSpaceMatrix);
    void endShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer);
    // Render a single object to shadow map
    // (use the VulkanApp pointer overload declared in the debug helpers section below)
    // Getters for resources
    VkImageView getShadowMapView() const { return shadowMapView; }
    VkSampler getShadowMapSampler() const { return shadowMapSampler; }
    VkDescriptorSet getImGuiDescriptorSet() const { return shadowMapImGuiDescSet; }
    uint32_t getShadowMapSize() const { return shadowMapSize; }
    VkDescriptorSetLayout getShadowDescriptorSetLayout(VulkanApp* app) const;
    // debug: read back depth image to host and write PGM
    // (use the VulkanApp pointer overload declared in the debug helpers section below)
    // Public getters for internal Vulkan handles
    VkRenderPass getShadowRenderPass() const { return shadowRenderPass; }
    VkFramebuffer getShadowFramebuffer() const { return shadowFramebuffer; }
    // Public getters for internal Vulkan handles
private:
    uint32_t shadowMapSize;
    // Shadow map resources
    VkImage shadowMapImage = VK_NULL_HANDLE;
    VkDeviceMemory shadowMapMemory = VK_NULL_HANDLE;
    VkImageView shadowMapView = VK_NULL_HANDLE;
    // Color attachment to make renderpass compatible with main pass
    VkImage shadowColorImage = VK_NULL_HANDLE;
    VkDeviceMemory shadowColorImageMemory = VK_NULL_HANDLE;
    VkImageView shadowColorImageView = VK_NULL_HANDLE;
    VkSampler shadowMapSampler = VK_NULL_HANDLE;
    VkFramebuffer shadowFramebuffer = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkPipeline shadowPipelineWire = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSet shadowMapImGuiDescSet = VK_NULL_HANDLE;
    
    // Current light space matrix for rendering
    glm::mat4 currentLightSpaceMatrix;
    // One-shot wireframe readback control
    bool requestWireframeReadbackFlag = false;
    bool performingWireframeReadback = false;
    
    void createShadowMap(VulkanApp* app);
    void createShadowRenderPass(VulkanApp* app);
    void createShadowFramebuffer(VulkanApp* app);
    void createShadowPipeline(VulkanApp* app);
    // Request next shadow pass be rendered in wireframe and read back
    void requestWireframeReadback();
    // Debug helpers
    void render(VulkanApp* app, VkCommandBuffer commandBuffer, 
                      const VertexBufferObject& vbo, VkDescriptorSet descriptorSet);
    void readbackShadowDepth(VulkanApp* app);
};
