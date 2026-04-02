#pragma once

#include "VulkanApp.hpp"
#include "VertexBufferObject.hpp"
#include "SkySphere.hpp"
#include "VertexBufferObjectBuilder.hpp"
#include "../math/SphereModel.hpp"
#include "ShaderStage.hpp"
#include "../Uniforms.hpp"
#include "../widgets/SkySettings.hpp"
#include <array>

class SkyRenderer {
public:
    explicit SkyRenderer();
    ~SkyRenderer();

    // Create sky pipelines; renderPassOverride lets callers supply a compatible render pass
    void init(VulkanApp* app, VkRenderPass renderPassOverride = VK_NULL_HANDLE);

    // Render sky sphere using internal VBO/descriptor/uniform
    void render(VulkanApp* app, VkCommandBuffer &cmd, VkDescriptorSet descriptorSet, Buffer &uniformBuffer, const UniformObject &ubo, const glm::mat4 &viewProjection, SkySettings::Mode skyMode);

    // Initialize the sky sphere and internal VBO (optional)
    void init(VulkanApp* app, SkySettings& skySettings, VkDescriptorSet descriptorSet);

    // Update sky internals (e.g. SkySphere animation)
    void update(VulkanApp* app);

    void cleanup();

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
    VkImageView getSkyView(uint32_t frameIndex) const { return skyColorImageViews[frameIndex]; }
    VkRenderPass getSkyRenderPass() const { return skyOffscreenRenderPass; }

private:
    VkPipeline skyPipeline = VK_NULL_HANDLE;        // Gradient sky pipeline
    VkPipelineLayout skyPipelineLayout = VK_NULL_HANDLE;
    VkPipeline skyGridPipeline = VK_NULL_HANDLE;   // Grid sky pipeline
    VkPipelineLayout skyGridPipelineLayout = VK_NULL_HANDLE;
    VkShaderModule skyVertModule = VK_NULL_HANDLE;
    VkShaderModule skyFragModule = VK_NULL_HANDLE;
    VkShaderModule skyGridFragModule = VK_NULL_HANDLE;
    // Owned sky sphere and VBO
    std::unique_ptr<SkySphere> skySphere;
    VertexBufferObject skyVBO;

    // --- Offscreen equirectangular sky resources (2 frames in flight) ---
    VkRenderPass skyOffscreenRenderPass = VK_NULL_HANDLE;
    std::array<VkImage, 2> skyColorImages = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkDeviceMemory, 2> skyColorMemories = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkImageView, 2> skyColorImageViews = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    std::array<VkFramebuffer, 2> skyFramebuffers = {VK_NULL_HANDLE, VK_NULL_HANDLE};

    // Equirect pipeline (fullscreen triangle, no vertex input, no depth)
    VkPipeline skyEquirectPipeline = VK_NULL_HANDLE;
    VkPipelineLayout skyEquirectPipelineLayout = VK_NULL_HANDLE;
    VkShaderModule skyEquirectVertModule = VK_NULL_HANDLE;
    VkShaderModule skyEquirectFragModule = VK_NULL_HANDLE;

    uint32_t offscreenWidth = 0;
    uint32_t offscreenHeight = 0;
};
