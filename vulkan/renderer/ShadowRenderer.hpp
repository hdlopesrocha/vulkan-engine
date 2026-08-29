#pragma once

#include "Renderer.hpp"
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

class ShadowRenderer : public Renderer {
public:
    ShadowRenderer(uint32_t maxShadowMapSize = 2048);
    ~ShadowRenderer();
    void init(VulkanApp* app);
    void cleanup(VulkanApp* app) override;

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
    void render(VulkanApp* app, VkCommandBuffer commandBuffer, uint32_t frameIdx,
                          Buffer& mainUniformBuffer, const UniformObject& uboStatic,
                          bool shadowsEnabled, bool renderSolid, bool vegetationEnabled,
                          bool shadowTessellationEnabled, float lodBias,
                          const glm::vec3& cameraPos, int maxTargetLod = 16);

    // Parallel shadow pass: each of the SHADOW_CASCADE_COUNT cascades is
    // rasterized on its own command buffer submitted to a distinct graphics
    // queue (app->getCubeQueue), so the 3 cascade depth draws overlap. The
    // cascade cull stays serial (shared scratch buffers) in a cull CB that
    // signals a per-frame semaphore each raster waits on; the EVSM blur is
    // serial (shares a single blurTemp image) and, together with the
    // main-camera cull restore, runs in a final CB that raises finalSignals
    // once the shadow map is fully ready. Each cascade uses its own UBO slot
    // (shadowUBO_) and descriptor set (shadowCascadeSets_) so they never
    // serialize on the shared main UBO. `waitSemaphore` gates the cull CB
    // (the shadow task's own cull-result semaphore); `finalSignals` are the
    // per-consumer semaphores (semShadow + veg/solid/water/360) raised at the
    // end so downstream passes wait on a complete shadow map.
    void renderParallel(VulkanApp* app, uint32_t frameIdx,
                          Buffer& mainUniformBuffer, const UniformObject& uboStatic,
                          bool shadowsEnabled, bool renderSolid, bool vegetationEnabled,
                          bool shadowTessellationEnabled, float lodBias,
                          const glm::vec3& cameraPos, int maxTargetLod,
                          VkSemaphore waitSemaphore,
                          const std::vector<VkSemaphore>& finalSignals);
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
    void setCmdState(CommandBufferState* state) override { Renderer::setCmdState(state); }
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

    // Per-frame staging buffers for UBO uploads via vkCmdCopyBuffer
    std::vector<Buffer> uboStagingBuffers_;

    // Per-frame shadow descriptor sets (cached handles, owned by SceneRenderer)
    std::vector<VkDescriptorSet> shadowDescriptorSets_;

    // ── Parallel per-cascade shadow resources ──
    // Each cascade gets its own UBO slot + descriptor set so the 3 cascade
    // rasterizations can overlap on independent command buffers / queues
    // instead of serializing on the shared main UBO (the old serial path
    // overwrote the main UBO per cascade). Built lazily once SceneRenderer
    // has populated shadowDescriptorSets_ (see ensureShadowParallelResources).
    std::vector<Buffer> shadowUBO_;  // per frame, SHADOW_CASCADE_COUNT UBO slots
    std::vector<std::array<VkDescriptorSet, SHADOW_CASCADE_COUNT>> shadowCascadeSets_;
    VkDescriptorPool cascadeDescPool_ = VK_NULL_HANDLE;
    bool cascadeSetsBuilt_ = false;

    // Per-frame internal semaphores for the parallel cascade graph:
    //   cullDone -> cascade[0..2] -> blurDone
    // Sized to MAX_FRAMES_IN_FLIGHT (3) so overlapping frames each own a
    // distinct set (binary semaphores must not be waited by two CBs).
    static constexpr uint32_t kShadowFrameSlots = 3; // == VulkanApp::MAX_FRAMES_IN_FLIGHT
    // One cull-done semaphore PER CASCADE per frame: the serial cull CB raises
    // all of them, and exactly one parallel cascade CB waits on its own slot. A
    // single binary semaphore may have only one waiter, so we cannot have all
    // three cascade CBs wait on the same semaphore.
    std::array<std::array<VkSemaphore, SHADOW_CASCADE_COUNT>, kShadowFrameSlots> semCullDone_ = {};
    std::array<std::array<VkSemaphore, SHADOW_CASCADE_COUNT>, kShadowFrameSlots> semCascadeDone_ = {};

    // Allocate (once) the per-cascade UBO / descriptor sets / semaphores used
    // by renderParallel. No-op until SceneRenderer has provided the shared
    // shadow descriptor sets; safe to call every frame.
    void ensureShadowParallelResources(VulkanApp* app);

    // Record a single cascade's depth draw into cmd: per-cascade UBO upload to
    // shadowUBO_ slot cascadeIndex, shadow pass begin/draw/end using the
    // cascade-specific descriptor set. frameSlot selects the per-frame UBO /
    // semaphore set.
    void recordCascade(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIdx,
                       const UniformObject& uboStatic, const glm::mat4& lsMatrix,
                       uint32_t cascadeIndex, uint32_t frameSlot,
                       bool renderSolid, bool vegetationEnabled,
                       bool shadowTessellationEnabled, float lodBias,
                       const glm::vec3& cameraPos);

    // Scene sub-renderers drawn into the shadow map (injected via setSceneRenderers)
    SolidRenderer* solidRenderer_ = nullptr;
    WaterRenderer* liquidRenderer_ = nullptr;
    VegetationRenderer* vegetationRenderer_ = nullptr;
    BrushRenderer* brushRenderer_ = nullptr;
};
