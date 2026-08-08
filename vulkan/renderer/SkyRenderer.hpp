#pragma once

#include "Renderer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../VertexBufferObject.hpp"
#include "../SkySphere.hpp"
#include "../VertexBufferObjectBuilder.hpp"
#include "../../math/SphereModel.hpp"
#include "../ShaderStage.hpp"
#include "../ubo/UniformObject.hpp"
#include "../../widgets/SkySettings.hpp"
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

    // Accessors for external renderers (cubemap 360 uses fullscreen sky pipeline)
    VkPipeline getSkyFullscreenPipeline() const { return skyFullscreenPipeline; }
    VkPipelineLayout getSkyFullscreenPipelineLayout() const { return skyFullscreenPipelineLayout; }
    VkPipeline getSkyFullscreenGridPipeline() const { return skyFullscreenGridPipeline; }
    VkPipelineLayout getSkyFullscreenGridPipelineLayout() const { return skyFullscreenGridPipelineLayout; }
    Buffer getSkyUniformBuffer() const;

private:
    // Sphere pipelines (kept for Solid360Renderer cubemap capture)
    TrackedHandle<VkPipeline> skyPipeline;
    TrackedHandle<VkPipelineLayout> skyPipelineLayout;
    TrackedHandle<VkPipeline> skyGridPipeline;
    TrackedHandle<VkPipelineLayout> skyGridPipelineLayout;
    TrackedHandle<VkShaderModule> skyVertModule;
    TrackedHandle<VkShaderModule> skyFragModule;
    TrackedHandle<VkShaderModule> skyGridFragModule;

    // Fullscreen pipelines (used for on-screen sky — no vertex input, 3 vertices total)
    TrackedHandle<VkPipeline> skyFullscreenPipeline;
    TrackedHandle<VkPipelineLayout> skyFullscreenPipelineLayout;
    TrackedHandle<VkPipeline> skyFullscreenGridPipeline;
    TrackedHandle<VkPipelineLayout> skyFullscreenGridPipelineLayout;
    TrackedHandle<VkShaderModule> skyFullscreenVertModule;

    // Owned sky sphere and VBO (retained for Solid360Renderer)
    std::unique_ptr<SkySphere> skySphere;
    VertexBufferObject skyVBO;

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
