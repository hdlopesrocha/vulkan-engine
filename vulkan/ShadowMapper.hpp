#pragma once

#include "vulkan.hpp"

class ShadowMapper {
public:
    ShadowMapper(VulkanApp* app, uint32_t shadowMapSize = 2048);
    ~ShadowMapper();
    
    void init();
    void cleanup();
    
    // Render shadow pass for a collection of objects
    void beginShadowPass(VkCommandBuffer commandBuffer, const glm::mat4& lightSpaceMatrix);
    void endShadowPass(VkCommandBuffer commandBuffer);
    
    // Render a single object to shadow map
    void renderObject(VkCommandBuffer commandBuffer, const glm::mat4& modelMatrix, 
                      const VertexBufferObject& vbo, VkDescriptorSet descriptorSet,
                      const glm::vec4& pomParams, const glm::vec4& pomFlags, const glm::vec3& viewPos);
    
    // Getters for resources
    VkImageView getShadowMapView() const { return shadowMapView; }
    VkSampler getShadowMapSampler() const { return shadowMapSampler; }
    VkDescriptorSet getImGuiDescriptorSet() const { return shadowMapImGuiDescSet; }
    uint32_t getShadowMapSize() const { return shadowMapSize; }
    
private:
    VulkanApp* vulkanApp;
    uint32_t shadowMapSize;
    
    // Shadow map resources
    VkImage shadowMapImage = VK_NULL_HANDLE;
    VkDeviceMemory shadowMapMemory = VK_NULL_HANDLE;
    VkImageView shadowMapView = VK_NULL_HANDLE;
    VkSampler shadowMapSampler = VK_NULL_HANDLE;
    VkFramebuffer shadowFramebuffer = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout shadowDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet shadowMapImGuiDescSet = VK_NULL_HANDLE;
    
    // Current light space matrix for rendering
    glm::mat4 currentLightSpaceMatrix;
    
    void createShadowMap();
    void createShadowRenderPass();
    void createShadowFramebuffer();
    void createShadowPipeline();
};
