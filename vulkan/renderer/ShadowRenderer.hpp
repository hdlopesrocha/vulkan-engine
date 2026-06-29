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
    // EVSM blur for a single cascade (horizontal + vertical passes)
    void blurCascade(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex);
    // Getters for resources (per-cascade)
    VkImageView getShadowMapView(uint32_t cascade = 0) const { return cascades[cascade].colorView; }
    VkSampler getShadowMapSampler() const { return shadowMapSampler; }
    VkDescriptorSet getImGuiDescriptorSet(uint32_t cascade = 0) const { return cascades[cascade].imguiDescSet; }
    uint32_t getShadowMapSize() const { return shadowMapSize; }
    VkDescriptorSetLayout getShadowDescriptorSetLayout(VulkanApp* app) const;
    VkRenderPass getShadowRenderPass() const { return VK_NULL_HANDLE; }
    VkFramebuffer getShadowFramebuffer(uint32_t cascade = 0) const { return VK_NULL_HANDLE; }
    VkPipeline getShadowPipeline() const { return shadowPipeline; }
    VkPipelineLayout getShadowPipelineLayout() const { return shadowPipelineLayout; }
    VkImageView getDummyDepthView() const { return dummyColorView; }
    VkImage getDepthImage(uint32_t cascade = 0) const;
    VkImageLayout getDepthLayout(uint32_t cascade = 0) const;
    void setDepthLayout(uint32_t cascade, VkImageLayout layout);
    void freeImGuiDescriptors();
    void recreateImGuiDescriptors();
private:
    uint32_t shadowMapSize;

    // Per-cascade resources (EVSM color image + depth image for depth testing)
    struct CascadeResources {
        // EVSM moments (RGBA32F)
        VkImage colorImage = VK_NULL_HANDLE;
        VkDeviceMemory colorMemory = VK_NULL_HANDLE;
        VkImageView colorView = VK_NULL_HANDLE;
        // Depth buffer for depth testing during shadow rendering
        VkImage depthImage = VK_NULL_HANDLE;
        VkDeviceMemory depthMemory = VK_NULL_HANDLE;
        VkImageView depthView = VK_NULL_HANDLE;
        VkDescriptorSet imguiDescSet = VK_NULL_HANDLE;
    };
    CascadeResources cascades[SHADOW_CASCADE_COUNT];

    // Shared sampler for all cascades (LINEAR filtering for EVSM)
    VkSampler shadowMapSampler = VK_NULL_HANDLE;

    // Dummy 1x1 RGBA32F image kept in SHADER_READ_ONLY layout for shadow pass descriptor set bindings
    VkImage dummyColorImage = VK_NULL_HANDLE;
    VkDeviceMemory dummyColorMemory = VK_NULL_HANDLE;
    VkImageView dummyColorView = VK_NULL_HANDLE;

    // Shadow pipeline (writes EVSM moments to color)
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;

    // Blur resources
    VkPipeline blurPipeline = VK_NULL_HANDLE;
    VkPipelineLayout blurPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout blurDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool blurDescPool = VK_NULL_HANDLE;
    // One descriptor set per cascade for horizontal blur (reads cascade color image)
    // + one shared for vertical blur (reads blurTemp)
    VkDescriptorSet blurHorizontalDS[SHADOW_CASCADE_COUNT] = {};
    VkDescriptorSet blurVerticalDS = VK_NULL_HANDLE;
    // Temporary image for separable blur ping-pong
    VkImage blurTempImage = VK_NULL_HANDLE;
    VkDeviceMemory blurTempMemory = VK_NULL_HANDLE;
    VkImageView blurTempView = VK_NULL_HANDLE;

    glm::mat4 currentLightSpaceMatrix;
    bool requestWireframeReadbackFlag = false;
    bool performingWireframeReadback = false;

    void createShadowMaps(VulkanApp* app);
    void createShadowPipeline(VulkanApp* app);
    void createBlurResources(VulkanApp* app);
    void requestWireframeReadback();
    void render(VulkanApp* app, VkCommandBuffer commandBuffer,
                      const VertexBufferObject& vbo, VkDescriptorSet descriptorSet);
    std::array<VkImageLayout, SHADOW_CASCADE_COUNT> cascadeDepthLayouts = {};
};
