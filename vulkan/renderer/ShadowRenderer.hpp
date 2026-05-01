#pragma once

#include "../vulkan.hpp"
#include <array>
#include "../ubo/UniformObject.hpp"

class ShadowRenderer {
public:
    ShadowRenderer(uint32_t shadowMapSize = 2048);
    ~ShadowRenderer();
    void init(VulkanApp* app);
    void cleanup(VulkanApp* app);
    // Render shadow pass for a single cascade
    void beginShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex, const glm::mat4& lightSpaceMatrix);
    void endShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex);
    // Getters for resources (per-cascade)
    VkImageView getShadowMapView(uint32_t cascade = 0) const { return cascades[cascade].depthView; }
    VkSampler getShadowMapSampler() const { return shadowMapSampler; }
    VkDescriptorSet getImGuiDescriptorSet(uint32_t cascade = 0) const { return cascades[cascade].imguiDescSet; }
    uint32_t getShadowMapSize() const { return shadowMapSize; }
    VkDescriptorSetLayout getShadowDescriptorSetLayout(VulkanApp* app) const;
    // Public getters for internal Vulkan handles
    VkRenderPass getShadowRenderPass() const { return shadowRenderPass; }
    VkFramebuffer getShadowFramebuffer(uint32_t cascade = 0) const { return cascades[cascade].framebuffer; }
    VkPipeline getShadowPipeline() const { return shadowPipeline; }
    VkPipelineLayout getShadowPipelineLayout() const { return shadowPipelineLayout; }
    VkImageView getDummyDepthView() const { return dummyDepthView; }
    // Expose raw depth image for readback/debugging
    VkImage getDepthImage(uint32_t cascade = 0) const;
    // Expose tracked layout for each cascade (used by debug widgets)
    VkImageLayout getDepthLayout(uint32_t cascade = 0) const;
private:
    uint32_t shadowMapSize;

    // Per-cascade resources (depth image, color image, framebuffer, ImGui descriptor)
    struct CascadeResources {
        VkImage depthImage = VK_NULL_HANDLE;
        VkDeviceMemory depthMemory = VK_NULL_HANDLE;
        VkImageView depthView = VK_NULL_HANDLE;
        VkImage colorImage = VK_NULL_HANDLE;
        VkDeviceMemory colorMemory = VK_NULL_HANDLE;
        VkImageView colorView = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkDescriptorSet imguiDescSet = VK_NULL_HANDLE;
    };
    CascadeResources cascades[SHADOW_CASCADE_COUNT];

    // Shared sampler for all cascades
    VkSampler shadowMapSampler = VK_NULL_HANDLE;
    // Dummy 1x1 depth image kept in READ_ONLY layout for shadow descriptor set bindings
    VkImage dummyDepthImage = VK_NULL_HANDLE;
    VkDeviceMemory dummyDepthMemory = VK_NULL_HANDLE;
    VkImageView dummyDepthView = VK_NULL_HANDLE;
    // Shared render pass and pipeline (same for all cascades, different framebuffer)
    VkRenderPass shadowRenderPass = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkPipeline shadowPipelineWire = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
    
    // Current light space matrix for rendering
    glm::mat4 currentLightSpaceMatrix;
    // One-shot wireframe readback control
    bool requestWireframeReadbackFlag = false;
    bool performingWireframeReadback = false;
    
    void createShadowMaps(VulkanApp* app);
    void createShadowRenderPass(VulkanApp* app);
    void createShadowFramebuffers(VulkanApp* app);
    void createShadowPipeline(VulkanApp* app);
    // Request next shadow pass be rendered in wireframe and read back
    void requestWireframeReadback();
    // Debug helpers
    void render(VulkanApp* app, VkCommandBuffer commandBuffer, 
                      const VertexBufferObject& vbo, VkDescriptorSet descriptorSet);
    // Track per-cascade depth image layouts for external callers
    std::array<VkImageLayout, SHADOW_CASCADE_COUNT> cascadeDepthLayouts = {};
};
