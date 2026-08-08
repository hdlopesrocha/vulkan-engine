#pragma once
#include "../VulkanApp.hpp"
#include "IndirectRenderer.hpp"
#include "BrushBackFaceRenderer.hpp"
#include "../../space/Octree.hpp"
#include "../../space/Model3DVersion.hpp"
#include "../../space/ThreadPool.hpp"
#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>

class SolidRenderer;

// Owns everything the brush preview needs on the GPU, mirroring how
// SolidRenderer / WaterRenderer encapsulate their offscreen targets, pipelines
// and IndirectRenderers:
//   - brush offscreen color + front depth targets (one per frame-in-flight)
//   - per-frame brush depth descriptor sets (set=1, bindings 0/1) used by the
//     solid/water shaders to depth-test against the brush volume
//   - the back-face depth pass (BrushBackFaceRenderer) for PAINT mode
//   - the two dedicated brush IndirectRenderers (solid + liquid), so the brush
//     geometry never shares the main scene slot pools
//   - the brush chunk registries (slot bookkeeping for erasure) and the
//     generation pools the brush octree change handlers tessellate on
// The early brush pass itself (front depth -> backface -> color, followed by
// SHADER_READ_ONLY transitions) is recorded by recordEarlyPass().
class BrushRenderer {
public:
    BrushRenderer();
    ~BrushRenderer();

    // Create everything brush-related (offscreen targets, back-face renderer,
    // descriptor sets, IndirectRenderers). Samplers for the brush depth
    // descriptor writes must be provided via setDepthSamplers() first (they
    // are queried lazily by writeDepthDescriptors()).
    void init(VulkanApp* app, uint32_t width, uint32_t height);
    void cleanup(VulkanApp* app);

    // Offscreen brush targets (color + front depth), matching MAX_FRAMES_IN_FLIGHT
    static constexpr uint32_t BRUSH_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);
    void onSwapchainResized(VulkanApp* app, uint32_t width, uint32_t height);

    VkImageView getColorView(uint32_t i) const { return colorImageViews[i % BRUSH_FRAMES]; }
    VkImage getColorImage(uint32_t i) const { return colorImages[i % BRUSH_FRAMES]; }
    VkImageView getDepthView(uint32_t i) const { return depthImageViews[i % BRUSH_FRAMES]; }
    VkImage getDepthImage(uint32_t i) const { return depthImages[i % BRUSH_FRAMES]; }
    VkImageView getBackFaceDepthView(uint32_t i) const {
        return backFaceRenderer ? backFaceRenderer->getBackFaceDepthView(i) : VK_NULL_HANDLE;
    }

    // Per-frame descriptor sets for brush depth textures (set=1, binding 0/1:
    // front depth, back-face depth).
    VkDescriptorSet getDepthDescriptorSet(uint32_t frameIndex) const {
        return depthDescriptorSets[frameIndex % BRUSH_FRAMES];
    }

    // Samplers used for the brush depth descriptor writes (queried each time
    // writeDepthDescriptors() runs, so they may be (re)created later).
    void setDepthSamplers(VkSampler linear, VkSampler shadow) {
        depthLinearSampler = linear;
        depthShadowSampler = shadow;
    }

    // Rewrite brush depth texture descriptors (bindings 0, 1) for all per-frame
    // brush descriptor sets. Must be called after brush render targets are
    // recreated (on resize or after init).
    void writeDepthDescriptors(VulkanApp* app);

    // Dedicated IndirectRenderers for the brush solid and liquid meshes (own
    // slot pools; never shared with the main scene IRs).
    IndirectRenderer& getSolidIR() { return solidIndirectRenderer; }
    IndirectRenderer& getLiquidIR() { return liquidIndirectRenderer; }
    void pollPendingTransfers(VulkanApp* app);
    void setCullFrame(uint32_t frameIndex);

    // Brush scene chunk tracking (separate from main scene). Public because the
    // space-change lambdas built by main.cpp read/write them directly.
    std::unordered_map<NodeID, Model3DVersion> solidChunks;
    std::unordered_map<NodeID, Model3DVersion> transparentChunks;
    std::unordered_map<NodeID, Model3DVersion> pendingOldSolidChunks;
    std::unordered_map<NodeID, Model3DVersion> pendingOldLiquidChunks;
    std::recursive_mutex solidChunksMutex;
    std::recursive_mutex transparentChunksMutex;
    std::recursive_mutex pendingOldSolidChunksMutex;
    std::recursive_mutex pendingOldLiquidChunksMutex;

    // Remove all brush meshes from GPU and clear the brush chunk maps.
    // (Draining the SHARED pending mesh queue of isBrush entries stays with
    // SceneRenderer, which owns that queue.)
    void clearMeshes();

    // Move the current chunk registries into the pending-old maps so the slots
    // survive until the new brush uploads complete (no 1-frame hole).
    void stageOldChunks();

    // Capture (and clear) the staged old slots for the current rebuild. The
    // caller feeds them into the shared publish core, which frees each old
    // slot after the replacement upload completes.
    void captureOldSlots(std::unordered_map<NodeID, uint32_t>& outSolid,
                         std::unordered_map<NodeID, uint32_t>& outTransparent);

    // Record the early brush pass into `cmd`: front depth (LESS, no color),
    // back-face depth (GREATER), brush color (raw), then transition all brush
    // targets to SHADER_READ_ONLY_OPTIMAL for the deferred composite.
    void recordEarlyPass(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameIndex,
                         SolidRenderer& solidRenderer, VkDescriptorSet mainDs);

    // Initialize the brush slot pools (packed element pools; the byte budgets
    // are TOTAL shared pool budgets, not per-chunk).
    void initSlots(VulkanApp* app, uint32_t maxChunks, uint32_t vertexBytes, uint32_t indexBytes);

    // Dedicated generation pools for brush solid and liquid so both layers
    // tessellate in parallel and never compete with the main scene pools.
    ThreadPool solidGenPool{std::max(2u, std::thread::hardware_concurrency() / 2)};
    ThreadPool liquidGenPool{std::max(2u, std::thread::hardware_concurrency() / 2)};
    void stopGenPools();

    // Back-face depth pass for the brush solid geometry (GREATER compare).
    std::unique_ptr<BrushBackFaceRenderer> backFaceRenderer;

private:
    // Offscreen brush render targets (color + front depth), one per frame
    std::array<VkImage, BRUSH_FRAMES> colorImages = {};
    std::array<VmaAllocation, BRUSH_FRAMES> colorAllocations = {};
    std::array<VkImageView, BRUSH_FRAMES> colorImageViews = {};
    std::array<VkImageLayout, BRUSH_FRAMES> colorLayouts = {};
    std::array<VkImage, BRUSH_FRAMES> depthImages = {};
    std::array<VmaAllocation, BRUSH_FRAMES> depthAllocations = {};
    std::array<VkImageView, BRUSH_FRAMES> depthImageViews = {};
    std::array<VkImageLayout, BRUSH_FRAMES> depthLayouts = {};

    // Per-frame descriptor sets for brush depth textures (set=1)
    std::array<VkDescriptorSet, BRUSH_FRAMES> depthDescriptorSets = {};

    IndirectRenderer solidIndirectRenderer;
    IndirectRenderer liquidIndirectRenderer;

    VkSampler depthLinearSampler = VK_NULL_HANDLE;
    VkSampler depthShadowSampler = VK_NULL_HANDLE;
};
