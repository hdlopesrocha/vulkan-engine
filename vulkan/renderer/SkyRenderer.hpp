#pragma once

#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../SkySphere.hpp"
#include "../ShaderStage.hpp"
#include "../ubo/UniformObject.hpp"
#include "../../widgets/SkySettings.hpp"
#include "../../widgets/CloudSettings.hpp"
#include <array>
#include "CommandBufferState.hpp"

class SkyRenderer : public Renderer {
public:
    explicit SkyRenderer();
    ~SkyRenderer();

    // Create sky pipelines using dynamic rendering
    void init(VulkanApp* app);

    // Render sky sphere using internal VBO/descriptor/uniform
    void render(VulkanApp* app, VkCommandBuffer &cmd, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, const UniformObject &ubo, const glm::mat4 &viewProjection, SkySettings::Mode skyMode);

    // Initialize the sky sphere and internal VBO (optional)
    void init(VulkanApp* app, SkySettings& skySettings, VkDescriptorSet descriptorSet);
    // Attach cloud settings (shares the SkyUBO) + animation clock.
    void setCloudSettings(CloudSettings* clouds);
    void setCloudTime(float t);
    void setCloudsEnabled(bool e);

    // Update sky internals (e.g. SkySphere animation)
    void update(VulkanApp* app);

    void cleanup(VulkanApp* app) override;

    // --- Offscreen sky rendering ---
    // Create offscreen render targets (color + depth, 2 frames in flight)
    void createOffscreenTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyOffscreenTargets(VulkanApp* app);

    // Render the sky to its own offscreen color attachment
    void renderOffscreen(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                         VkDescriptorSet descriptorSet, Buffer &uniformBuffer,
                         const UniformObject &ubo, const glm::mat4 &viewProjection,
                         SkySettings::Mode skyMode);

    // Access offscreen sky color view for sampling
    VkImageView getSkyView(uint32_t frameIndex) const {
        VkImageView v = skyColorImageViews[frameIndex % SKY_FRAMES];
        if (v != VK_NULL_HANDLE) return v;
        for (size_t i = skyColorImageViews.size(); i-- > 0; )
            if (skyColorImageViews[i] != VK_NULL_HANDLE) return skyColorImageViews[i];
        return VK_NULL_HANDLE;
    }

    // Sky UBO (binding 6) accessor — consumed by SceneRenderer for RT scene views.
    Buffer getSkyUniformBuffer() const;

private:
    TrackedHandle<VkShaderModule> skyFragModule;
    TrackedHandle<VkShaderModule> skyGridFragModule;

    // Fullscreen pipelines (used for on-screen sky — no vertex input, 3 vertices total)
    TrackedHandle<VkPipeline> skyFullscreenPipeline;
    TrackedHandle<VkPipelineLayout> skyFullscreenPipelineLayout;
    TrackedHandle<VkPipeline> skyFullscreenGridPipeline;
    TrackedHandle<VkPipelineLayout> skyFullscreenGridPipelineLayout;
    TrackedHandle<VkShaderModule> skyFullscreenVertModule;

    // SkySphere owns the dedicated SkyUBO (binding 6); consumed by
    // SceneRenderer for RT scene views. The on-screen/offscreen sky passes are
    // fullscreen triangles with no sphere geometry (the sphere pipelines and
    // VBO were removed with Solid360Renderer — perf report 22 M9).
    std::unique_ptr<SkySphere> skySphere;
    // Cloud attachment staged before SkySphere exists (init order).
    CloudSettings* pendingClouds = nullptr;
    float pendingCloudTime = 0.0f;
    bool pendingCloudsEnabled = true;
    bool hasPendingCloudsEnabled = false;

    // --- Offscreen equirectangular sky resources matching MAX_FRAMES_IN_FLIGHT ---
    static constexpr uint32_t SKY_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<VkImage, SKY_FRAMES> skyColorImages = {};
    std::array<VmaAllocation, SKY_FRAMES> skyColorAllocations = {};
    std::array<VkDeviceMemory, SKY_FRAMES> skyColorMemories = {};
    std::array<VkImageView, SKY_FRAMES> skyColorImageViews = {};
    std::array<VkImageLayout, SKY_FRAMES> skyColorLayouts = {};

    // Equirect pipeline (fullscreen triangle, no vertex input, no depth)
    TrackedHandle<VkPipeline> skyEquirectPipeline;
    TrackedHandle<VkPipelineLayout> skyEquirectPipelineLayout;
    TrackedHandle<VkShaderModule> skyEquirectVertModule;
    TrackedHandle<VkShaderModule> skyEquirectFragModule;

    uint32_t offscreenWidth = 0;
    uint32_t offscreenHeight = 0;
};
