#pragma once

#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../core/TrackedHandle.hpp"
#include "../../resources/TextureImage.hpp"
#include "../indirect/IndirectRenderer.hpp"
#include "../CommandBufferState.hpp"

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

class VideoTextureService;

// ── Video layer renderer ──────────────────────────────────────────────────
// Renders the video scene layer: octree geometry whose materials are
// MaterialKind::Video. The layer is drawn after solid and water (the composite
// order is Solid -> Water -> Video) into its own offscreen color+depth target.
//
// The color target stores the material albedo in rgb and the material blend
// mode in alpha, encoded as (mode + 1) / 32 with 0 meaning "no video". The
// post-process composite decodes it and applies the matching Photoshop-style
// blend mode (shaders/includes/BlendModes.glsl) against the already composed
// scene color, which is why the video layer needs a separate pass: a forward
// draw cannot read the destination it blends with.
//
// Geometry uses the engine's standard octree mesh format (IndirectRenderer),
// the non-tessellated solid vertex shader and a dedicated fragment shader that
// samples the video texture array through the material's textureIndex.
class VideoRenderer : public Renderer {
public:
    static constexpr uint32_t VIDEO_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;

    VideoRenderer();
    ~VideoRenderer() override;

    void init(VulkanApp* app);
    void createPipelines(VulkanApp* app);
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);
    void cleanup(VulkanApp* app) override;

    // Bind the video texture array (set 1). Pass nullptr to keep the dummy
    // 1x1 black texture (no video service: the layer clears to "no video").
    void setVideoTexture(VulkanApp* app, VideoTextureService* service);

    // Record the whole video pass: GPU-cull the layer's chunks, clear the
    // target, draw every visible mesh, transition the targets back to
    // SHADER_READ_ONLY.
    void draw(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex,
              VkDescriptorSet sceneSet, const glm::mat4& viewProjection,
              const glm::vec3& cameraPos, float lodBias, int maxTargetLod);

    // Clear-only variant (layer disabled / no video materials loaded).
    void clear(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex);

    IndirectRenderer& getIndirectRenderer() { return indirectRenderer; }

    VkImageView colorView(uint32_t frameIndex) const {
        return (frameIndex < VIDEO_FRAMES) ? colorViews[frameIndex] : VK_NULL_HANDLE;
    }
    VkImageView depthView(uint32_t frameIndex) const {
        return (frameIndex < VIDEO_FRAMES) ? depthViews[frameIndex] : VK_NULL_HANDLE;
    }
    VkImageLayout colorLayout(uint32_t frameIndex) const {
        return (frameIndex < VIDEO_FRAMES) ? colorLayouts[frameIndex] : VK_IMAGE_LAYOUT_UNDEFINED;
    }
    VkImageLayout depthLayout(uint32_t frameIndex) const {
        return (frameIndex < VIDEO_FRAMES) ? depthLayouts[frameIndex] : VK_IMAGE_LAYOUT_UNDEFINED;
    }
    uint32_t getRenderWidth() const { return renderWidth; }
    uint32_t getRenderHeight() const { return renderHeight; }

private:
    // Opens the dynamic-rendering scope with a cleared color (0,0,0,0) and
    // depth (1.0); endPass() closes it and restores the sampled layouts.
    bool beginPass(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex);
    void endPass(VkCommandBuffer cmd, VulkanApp* app, uint32_t frameIndex);
    bool createVideoDescriptorSet(VulkanApp* app);
    void writeVideoDescriptor(VulkanApp* app, VkImageView view, VkSampler sampler);

    IndirectRenderer indirectRenderer;
    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> videoSetLayout;
    TrackedHandle<VkDescriptorPool> videoSetPool;
    VkDescriptorSet videoSet = VK_NULL_HANDLE;
    VkImageView boundVideoView = VK_NULL_HANDLE;
    VkSampler boundVideoSampler = VK_NULL_HANDLE;

    // Dummy 1x1 black layer used until a real video array is bound.
    TextureImage dummyVideo;
    VkImageLayout dummyLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::array<VkImage, VIDEO_FRAMES> colorImages{};
    std::array<VmaAllocation, VIDEO_FRAMES> colorAllocations{};
    std::array<VkDeviceMemory, VIDEO_FRAMES> colorMemories{};
    std::array<VkImageView, VIDEO_FRAMES> colorViews{};
    std::array<VkImageLayout, VIDEO_FRAMES> colorLayouts{};

    std::array<VkImage, VIDEO_FRAMES> depthImages{};
    std::array<VmaAllocation, VIDEO_FRAMES> depthAllocations{};
    std::array<VkDeviceMemory, VIDEO_FRAMES> depthMemories{};
    std::array<VkImageView, VIDEO_FRAMES> depthViews{};
    std::array<VkImageLayout, VIDEO_FRAMES> depthLayouts{};

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
};
