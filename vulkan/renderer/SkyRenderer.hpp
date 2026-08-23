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

    // Initialize the sky sphere and internal VBO (optional)
    void init(VulkanApp* app, SkySettings& skySettings, VkDescriptorSet descriptorSet);

    // Update sky internals (e.g. SkySphere animation)
    void update(VulkanApp* app);

    void cleanup(VulkanApp* app) override;

    // --- Offscreen sky targets (kept; texture no longer raster-sampled) ---
    void createOffscreenTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyOffscreenTargets(VulkanApp* app);

    // Access offscreen sky color view for sampling
    VkImageView getSkyView(uint32_t frameIndex) const {
        VkImageView v = skyColorImageViews[frameIndex % SKY_FRAMES];
        if (v != VK_NULL_HANDLE) return v;
        for (size_t i = skyColorImageViews.size(); i-- > 0; )
            if (skyColorImageViews[i] != VK_NULL_HANDLE) return skyColorImageViews[i];
        return VK_NULL_HANDLE;
    }

    Buffer getSkyUniformBuffer() const;

private:
    // Owned sky sphere (holds the Sky UBO the ray tracer reads) and its VBO.
    std::unique_ptr<SkySphere> skySphere;
    VertexBufferObject skyVBO;

    // --- Offscreen sky colour targets (sampled by the present path / debug widget) ---
    static constexpr uint32_t SKY_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    std::array<VkImage, SKY_FRAMES> skyColorImages = {};
    std::array<VmaAllocation, SKY_FRAMES> skyColorAllocations = {};
    std::array<VkDeviceMemory, SKY_FRAMES> skyColorMemories = {};
    std::array<VkImageView, SKY_FRAMES> skyColorImageViews = {};
    std::array<VkImageLayout, SKY_FRAMES> skyColorLayouts = {};

    uint32_t offscreenWidth = 0;
    uint32_t offscreenHeight = 0;
};
