#pragma once

#include "../vulkan.hpp"
#include "../TrackedHandle.hpp"
#include "../Buffer.hpp"
#include <array>
#include <vector>
#include "../ubo/UniformObject.hpp"
#include "CommandBufferState.hpp"

class SolidRenderer;
class WaterRenderer;
class VegetationRenderer;
class BrushRenderer;

class ShadowRenderer {
public:
    ShadowRenderer(uint32_t maxShadowMapSize = 2048);
    ~ShadowRenderer();
    void init(VulkanApp* app);
    void cleanup(VulkanApp* app);

    // Inject the scene sub-renderers whose geometry is drawn into the shadow
    // map. Called once by SceneRenderer after all sub-renderers are created.
    void setSceneRenderers(SolidRenderer* solid, WaterRenderer* liquid,
                           VegetationRenderer* vegetation, BrushRenderer* brush);

    // Per-frame staging buffers for shadow UBO uploads via vkCmdCopyBuffer
    // (replaces vkCmdUpdateBuffer to avoid implicit FULL_QUEUE barrier).
    // Sized to hold SHADOW_CASCADE_COUNT cascade UBOs plus the restored main
    // UBO per frame slot.
    void createStagingBuffers(VulkanApp* app, size_t frameCount);
    void destroyStagingBuffers();

    // Shadow-specific descriptor sets (one per frame). Each mirrors the main
    // descriptor set but bindings 4, 8, 9 point to a dummy depth view. The
    // sets are allocated and maintained by SceneRenderer (which owns the
    // static scene descriptor wiring); their handles are stable after init,
    // so ShadowRenderer caches them once.
    void setShadowDescriptorSets(const std::vector<VkDescriptorSet>& sets) { shadowDescriptorSets_ = sets; }

    // Full shadow pass orchestration: per-cascade UBO upload, culling of the
    // solid/water/vegetation scene against all cascade frustums, cascade
    // draws, and restore of the main UBO + main-camera cull afterwards.
    // `cameraPos` must be the same position used for the main pass cull so
    // the cascade cull picks the identical per-chunk LoD selection.
    void renderShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t frameIdx,
                          Buffer& mainUniformBuffer, const UniformObject& uboStatic,
                          bool shadowsEnabled, bool renderSolid, bool vegetationEnabled,
                          bool shadowTessellationEnabled, float lodBias,
                          const glm::vec3& cameraPos);
    // Render shadow pass for a single cascade
    void beginShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex, const glm::mat4& lightSpaceMatrix);
    void endShadowPass(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex);
    // EVSM blur for a single cascade (horizontal + vertical passes)
    void blurCascade(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t cascadeIndex);
    // Getters for resources (per-cascade)
    VkImageView getShadowMapView(uint32_t cascade = 0) const { return cascades[cascade].colorView; }
    VkImageView getShadowDepthView(uint32_t cascade = 0) const { return cascades[cascade].depthView; }
    VkSampler getShadowMapSampler() const { return shadowMapSampler; }
    VkDescriptorSet getImGuiDescriptorSet(uint32_t cascade = 0) const { return cascades[cascade].imguiDescSet; }
    uint32_t getShadowMapSize(uint32_t cascade = 0) const { return shadowMapSizes[cascade]; }
    VkPipeline getShadowPipeline() const { return shadowPipeline; }
    VkPipelineLayout getShadowPipelineLayout() const { return shadowPipelineLayout; }
    VkImageView getDummyDepthView() const { return dummyColorView; }
    VkImage getDepthImage(uint32_t cascade = 0) const;
    VkImageLayout getDepthLayout(uint32_t cascade = 0) const;
    void setDepthLayout(uint32_t cascade, VkImageLayout layout);
    void freeImGuiDescriptors();
    void recreateImGuiDescriptors();
    void setCmdState(CommandBufferState* state) { cmdState = state; }
private:
    uint32_t shadowMapSizes[SHADOW_CASCADE_COUNT];

    // Per-cascade resources (EVSM color image + depth image for depth testing)
    struct CascadeResources {
        // EVSM moments (RGBA32F)
        VkImage colorImage = VK_NULL_HANDLE;
        VmaAllocation colorAllocation = VK_NULL_HANDLE;
        VkDeviceMemory colorMemory = VK_NULL_HANDLE;
        VkImageView colorView = VK_NULL_HANDLE;
        // Depth buffer for depth testing during shadow rendering
        VkImage depthImage = VK_NULL_HANDLE;
        VmaAllocation depthAllocation = VK_NULL_HANDLE;
        VkDeviceMemory depthMemory = VK_NULL_HANDLE;
        VkImageView depthView = VK_NULL_HANDLE;
        TrackedHandle<VkDescriptorSet> imguiDescSet;
    };
    CascadeResources cascades[SHADOW_CASCADE_COUNT];

    // Shared sampler for all cascades (LINEAR filtering for EVSM)
    TrackedHandle<VkSampler> shadowMapSampler;

    // Dummy 1x1 RGBA32F image kept in SHADER_READ_ONLY layout for shadow pass descriptor set bindings
    VkImage dummyColorImage = VK_NULL_HANDLE;
    VmaAllocation dummyColorAllocation = VK_NULL_HANDLE;
    VkDeviceMemory dummyColorMemory = VK_NULL_HANDLE;
    VkImageView dummyColorView = VK_NULL_HANDLE;

    // Shadow pipeline (writes EVSM moments to color)
    TrackedHandle<VkPipeline> shadowPipeline;
    TrackedHandle<VkPipelineLayout> shadowPipelineLayout;

    // Blur resources
    TrackedHandle<VkPipeline> blurPipeline;
    TrackedHandle<VkPipelineLayout> blurPipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> blurDescSetLayout;
    TrackedHandle<VkDescriptorPool> blurDescPool;
    // One descriptor set per cascade for horizontal blur (reads cascade color image)
    // + one shared for vertical blur (reads blurTemp)
    TrackedHandle<VkDescriptorSet> blurHorizontalDS[SHADOW_CASCADE_COUNT];
    TrackedHandle<VkDescriptorSet> blurVerticalDS;
    // Temporary image for separable blur ping-pong
    VkImage blurTempImage = VK_NULL_HANDLE;
    VmaAllocation blurTempAllocation = VK_NULL_HANDLE;
    VkDeviceMemory blurTempMemory = VK_NULL_HANDLE;
    VkImageView blurTempView = VK_NULL_HANDLE;

    void createShadowMaps(VulkanApp* app);
    void createShadowPipeline(VulkanApp* app);
    void createBlurResources(VulkanApp* app);
    std::array<VkImageLayout, SHADOW_CASCADE_COUNT> cascadeDepthLayouts = {};
    CommandBufferState* cmdState = nullptr;

    // Per-frame staging buffers for UBO uploads via vkCmdCopyBuffer
    std::vector<Buffer> uboStagingBuffers_;

    // Per-frame shadow descriptor sets (cached handles, owned by SceneRenderer)
    std::vector<VkDescriptorSet> shadowDescriptorSets_;

    // Scene sub-renderers drawn into the shadow map (injected via setSceneRenderers)
    SolidRenderer* solidRenderer_ = nullptr;
    WaterRenderer* liquidRenderer_ = nullptr;
    VegetationRenderer* vegetationRenderer_ = nullptr;
    BrushRenderer* brushRenderer_ = nullptr;
};
