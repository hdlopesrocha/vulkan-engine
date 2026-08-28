#pragma once

#include "Renderer.hpp"
#include "../Buffer.hpp"
#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../../math/BoundingCube.hpp"
#include "../../space/Octree.hpp" // for NodeID, Octree, OctreeNode
#include "../../space/OctreeNodeData.hpp"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "CommandBufferState.hpp"

class IndirectRenderer; // forward declaration: the solid IndirectRenderer owns the merged SDF cull dispatch

// Renders leaf-node cube faces colored by their SDF sign and magnitude.
class DebugSDFRenderer : public Renderer {
public:
    struct CubeSDF {
        BoundingCube cube;
        std::array<float, 8> sdf;
        int brushIndex;
        // LoD meta (cellSize/level/base) for the SDF cull's clipmap band gate.
        float cellSize = 0.0f;
        int level = 0;
        glm::vec3 base = glm::vec3(0.0f);
    };

    DebugSDFRenderer();
    ~DebugSDFRenderer();

    void init(VulkanApp* app);
    void setCubes(const std::vector<CubeSDF>& cubes);
    // The solid IndirectRenderer performs the SDF cube frustum cull + compaction in
    // its OWN indirect.comp dispatch (folded into the terrain cull). Point this
    // renderer at it so render() can draw from the terrain IR's SDF output buffers.
    void setIndirectRenderer(IndirectRenderer* ir) { terrainIR_ = ir; }
    // Upload the per-cube SDF instance payload for this frame. The frustum cull
    // itself is performed by the solid IndirectRenderer's indirect.comp dispatch
    // (folded into the terrain cull); render() draws from the terrain IR's SDF
    // output buffers. Call OUTSIDE a render pass, before the draw. `frame` must
    // match setCullFrame so upload + draw share one slot.
    void prepareCull(VkCommandBuffer cmd);
    void render(VulkanApp* app, VkCommandBuffer& cmd, VkDescriptorSet descriptorSet, uint32_t frameIdx, bool enabled = true);
    void setCullFrame(uint32_t frame) { currentCullFrame = frame % SDF_CULL_FRAMES; }
    void cleanup(VulkanApp* app) override;

    // ── Decoupled offscreen framebuffer ──
    // The SDF debug pass renders to its own color+depth offscreen (one per frame
    // in flight) so it can run on its own async command buffer / queue; the
    // composite (postprocess.frag) blends it over the solid scene by depth.
    static constexpr uint32_t SDF_FRAMES = VulkanApp::MAX_FRAMES_IN_FLIGHT;
    void createRenderTargets(VulkanApp* app, uint32_t width, uint32_t height);
    void destroyRenderTargets(VulkanApp* app);
    VkImageView getSdfColorView(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfColorImageViews[frameIndex] : VK_NULL_HANDLE; }
    VkImageView getSdfDepthView(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfDepthImageViews[frameIndex] : VK_NULL_HANDLE; }
    VkImage getSdfColorImage(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfColorImages[frameIndex] : VK_NULL_HANDLE; }
    VkImage getSdfDepthImage(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfDepthImages[frameIndex] : VK_NULL_HANDLE; }
    VkImageLayout getSdfColorLayout(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfColorImageLayouts[frameIndex] : VK_IMAGE_LAYOUT_UNDEFINED; }
    VkImageLayout getSdfDepthLayout(uint32_t frameIndex) const { return (frameIndex < SDF_FRAMES) ? sdfDepthImageLayouts[frameIndex] : VK_IMAGE_LAYOUT_UNDEFINED; }
    void setSdfColorLayout(uint32_t frameIndex, VkImageLayout l) { if (frameIndex < SDF_FRAMES) sdfColorImageLayouts[frameIndex] = l; }
    void setSdfDepthLayout(uint32_t frameIndex, VkImageLayout l) { if (frameIndex < SDF_FRAMES) sdfDepthImageLayouts[frameIndex] = l; }

    // ── Per-chunk SDF cube tracking (moved from SceneRenderer) ──
    // Populated by processNodeLayer via scene.requestSDFCubes (worker thread)
    // after chunk (re)builds; consumed as a copy on the render thread.
    void updateCubesForChunk(NodeID nid, const std::vector<CubeSDF>& cubes);
    void removeCubesForNode(NodeID id);
    void clearCubes();
    // Register the cached per-chunk SDF cubes with the solid IndirectRenderer's
    // merged cull dispatch (folds SDF AABBs into the terrain frustum cull). Reads
    // the cache internally — call OUTSIDE a render pass, before the solid IR's
    // prepareCull. Requires setIndirectRenderer to have been called.
    void registerToIndirect();

private:
    // Per-node SDF face cubes keyed by octree node id
    std::unordered_map<NodeID, std::vector<CubeSDF>> nodeDebugSDFCubes;
    mutable std::recursive_mutex cubesMutex;

    struct CubeVertex {
        glm::vec3 position;
        uint32_t cornerIndex;
    };

    // Per-cube GPU instance payload (matches debug_sdf.vert InstanceData).
    struct InstanceData {
        glm::mat4 model;
        glm::vec4 sdf0;
        glm::vec4 sdf1;
        glm::vec4 meta; // meta.x = brushIndex
    };

    static constexpr uint32_t SDF_CULL_FRAMES = 3;

    TrackedHandle<VkPipeline> pipeline;
    TrackedHandle<VkPipelineLayout> pipelineLayout;
    TrackedHandle<VkShaderModule> vertModule;
    TrackedHandle<VkShaderModule> fragModule;

    Buffer vertexBuffer;
    Buffer indexBuffer;
    uint32_t indexCount = 0;

    TrackedHandle<VkDescriptorSetLayout> descriptorSetLayout;
    TrackedHandle<VkDescriptorPool> descriptorPool;
    // One instance descriptor set per cull frame (matches the per-cull-frame
    // instance buffers in cullFrames[]). Each set points to its cull frame's
    // instance buffer and is written only when that buffer is (re)allocated, never
    // per-frame — so a set is never updated while a command buffer using it is
    // pending (the root cause of the vkUpdateDescriptorSets "in use by pending CB"
    // validation error). The binding carries UPDATE_AFTER_BIND_BIT as a safety net.
    std::array<VkDescriptorSet, SDF_CULL_FRAMES> sdfInstanceSets{};

    std::vector<CubeSDF> activeCubes;

    // ── Per-frame instance payload (read by the SDF vertex shader) ──
    // One set of buffers per cull frame (rotated) so an in-flight draw on the
    // same slot (3 frames in flight, FIFO) never races the host rewrite. The
    // visible DrawCmd stream + count live in the terrain IndirectRenderer's SDF
    // output buffers (folded into the terrain cull); only the per-cube instance
    // transform/value payload is stored here.
    struct CullFrame {
        Buffer instance;      // InstanceData per cube, indexed by gl_InstanceIndex
        uint32_t capacity = 0; // entries capacity
    };
    std::array<CullFrame, SDF_CULL_FRAMES> cullFrames;
    uint32_t currentCullFrame = 0;
    bool hasCubes_ = false;

    PFN_vkCmdDrawIndexedIndirectCountKHR cmdDrawIndexedIndirectCount = nullptr;
    VulkanApp* cullApp_ = nullptr; // stashed for buffer (re)allocation
    IndirectRenderer* terrainIR_ = nullptr; // owns the merged SDF cull dispatch

    // Offscreen color+depth targets (one per frame in flight).
    std::array<VkImage, SDF_FRAMES> sdfColorImages{};
    std::array<VmaAllocation, SDF_FRAMES> sdfColorAllocations{};
    std::array<VkDeviceMemory, SDF_FRAMES> sdfColorMemories{};
    std::array<VkImageView, SDF_FRAMES> sdfColorImageViews{};
    std::array<VkImageLayout, SDF_FRAMES> sdfColorImageLayouts{};
    std::array<VkImage, SDF_FRAMES> sdfDepthImages{};
    std::array<VmaAllocation, SDF_FRAMES> sdfDepthAllocations{};
    std::array<VkDeviceMemory, SDF_FRAMES> sdfDepthMemories{};
    std::array<VkImageView, SDF_FRAMES> sdfDepthImageViews{};
    std::array<VkImageLayout, SDF_FRAMES> sdfDepthImageLayouts{};
    uint32_t sdfRenderWidth = 0;
    uint32_t sdfRenderHeight = 0;

private:
    void createCubeBuffers(VulkanApp* app);
    void createDescriptorSet(VulkanApp* app);
    void createCullResources(VulkanApp* app);
    void ensureCullCapacity(uint32_t frame, uint32_t cubeCount);
    void writeCullFrameData(uint32_t frame, uint32_t cubeCount);
};
