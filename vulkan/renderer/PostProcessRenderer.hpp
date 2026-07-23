#pragma once

#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include <glm/glm.hpp>
#include <array>
#include "CommandBufferState.hpp"

// Forward-declare shared water types (defined in WaterRenderer.hpp)
struct WaterParams;
struct WaterUBO;

class PostProcessRenderer {
public:
    PostProcessRenderer();
    ~PostProcessRenderer();

    void init(VulkanApp* app);
    void cleanup(VulkanApp* app);

    /// Composite scene + water + brush into the swapchain framebuffer.
    /// Brush color/depth views come from the early brush pass offscreen targets.
    /// waterGeomDepthView is the raw water geometry depth buffer (D32).
    /// brushAlpha controls the brush overlay opacity (0.0 = invisible, 1.0 = fully opaque).
    void render(VulkanApp* app, VkCommandBuffer cmd,
                VkImageView sceneColorView, VkImageView sceneDepthView,
                VkImageView waterColorView,
                VkImageView brushColorView, VkImageView brushDepthView,
                VkImageView waterGeomDepthView,
                float brushAlpha,
                const glm::mat4& viewProj, const glm::mat4& invViewProj,
                const glm::vec3& viewPos,
                uint32_t frameIdx,
                VkImageView skyView = VK_NULL_HANDLE);

    bool isReady() const { return pipeline != VK_NULL_HANDLE; }

    VkSampler getLinearSampler() const { return linearSampler; }

    void setRenderSize(uint32_t width, uint32_t height);

private:
    void createSampler(VulkanApp* app);
    void createPipeline(VulkanApp* app);
    void createDescriptorSets(VulkanApp* app);

    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> descriptorSetLayout;
    TrackedHandle<VkDescriptorPool> descriptorPool;
    static constexpr uint32_t FRAMES_IN_FLIGHT = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<TrackedHandle<VkDescriptorSet>, FRAMES_IN_FLIGHT> descriptorSets;

    Buffer uniformBuffer;
    TrackedHandle<VkSampler> linearSampler;

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    CommandBufferState* cmdState = nullptr;
public:
    void setCmdState(CommandBufferState* state) { cmdState = state; }
};