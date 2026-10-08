#pragma once
#include "../Renderer.hpp"
#include "../../core/VulkanApp.hpp"
#include "../../ubo/BrushSdfUBO.hpp"
#include "../../../types/BrushEntry.hpp"
#include <array>

// SDF-driven brush preview renderer.
//
// Replaces the tessellated brush-scene raster pass entirely: the selected
// brush entry is uploaded as a small std140 UBO (set=1 binding=0) per frame
// and raymarched into an offscreen color + front depth pair by a fullscreen
// fragment pass. The pass shades hits through the same material arrays as the
// solid terrain shader (triplanar albedo/normal/roughness/AO + HSV tint), so
// the preview always matches the shape and material the brush will apply to
// the octree (the CPU apply path is unchanged).
//
// Consumers:
//   * PostProcess composites the color/depth targets over the scene.
//   * SolidRenderer binds the params descriptor set (set=1) so the solid
//     shader can evaluate the brush SDF directly for PAINT/REMOVE mode
//     (inside = tint override), replacing the old front/back depth textures.
class BrushSdfRenderer : public Renderer {
public:
    BrushSdfRenderer();
    ~BrushSdfRenderer();

    static constexpr uint32_t BRUSH_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;

    // Create offscreen targets, per-frame params buffers/descriptor sets and
    // the raymarch pipeline. Samplers for the depth descriptor writes are not
    // needed anymore (the params set carries a UBO).
    void init(VulkanApp* app, uint32_t width, uint32_t height);
    void cleanup(VulkanApp* app) override;

    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

    VkImageView getColorView(uint32_t i) const { return colorImageViews[i % BRUSH_FRAMES]; }
    VkImageView getDepthView(uint32_t i) const { return depthImageViews[i % BRUSH_FRAMES]; }

    // Per-frame brush description. `boundsCenter`/`boundsRadius` bound the
    // field (CPU-computed, effect-padded); radius <= 0 disables the preview
    // and the solid PAINT/REMOVE test. Must run on the main thread during
    // frame recording, before the brush pass and the solid draw.
    void updateParams(uint32_t frameIndex, const BrushEntry& entry,
                      const glm::vec3& sweepStart,
                      const glm::vec3& boundsCenter, float boundsRadius,
                      uint32_t width, uint32_t height);

    // Params descriptor set (set=1) consumed by both the preview pipeline and
    // SolidRenderer's draws.
    VkDescriptorSet getParamsDescriptorSet(uint32_t frameIndex) const {
        return paramsSets[frameIndex % BRUSH_FRAMES];
    }

    // Record the raymarch pass: transitions color/depth to attachments,
    // clears, draws the fullscreen triangle, transitions to
    // SHADER_READ_ONLY_OPTIMAL for the composite.
    void recordPass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                    VkDescriptorSet mainDs);

    void setCmdState(CommandBufferState* state) override { Renderer::setCmdState(state); }

private:
    std::array<VkImage, BRUSH_FRAMES> colorImages = {};
    std::array<VmaAllocation, BRUSH_FRAMES> colorAllocations = {};
    std::array<VkImageView, BRUSH_FRAMES> colorImageViews = {};
    std::array<VkImageLayout, BRUSH_FRAMES> colorLayouts = {};
    std::array<VkImage, BRUSH_FRAMES> depthImages = {};
    std::array<VmaAllocation, BRUSH_FRAMES> depthAllocations = {};
    std::array<VkImageView, BRUSH_FRAMES> depthImageViews = {};
    std::array<VkImageLayout, BRUSH_FRAMES> depthLayouts = {};

    std::array<Buffer, BRUSH_FRAMES> paramsBuffers = {};
    std::array<VkDescriptorSet, BRUSH_FRAMES> paramsSets = {};

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
};
